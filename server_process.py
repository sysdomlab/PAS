import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from threading import Lock
from math import log2

CUDA = torch.cuda.is_available()

CAS_hyperparams_profile = {
    'ema_theta' : 0.7,
    'observe_window_size' : 5
}

class CAS_AggregationService(Service):
    def __init__(self, world_size, init_period) -> None:
        super().__init__()
        self.flattened_params_length = None
        self.world_size = world_size
        self.averaged_tensor = None
        self.eam_theta = CAS_hyperparams_profile['ema_theta']
        self.lock = Lock()
        self.round_id = 0
        self.initial_period = init_period
        self.global_period = init_period

        self.param_sync_period = None
        self.last_tensor_received = None
        self.positive_grad = None
        self.negative_grad = None
        self.ema_positive_grad = None
        self.ema_negative_grad = None
        self.mask = None
        self.grad_consistency = None
        self.min_grad_consistency = None
        self.window_start_round = None
        self.global_tune_index = None
        self.observe_window_size = None
        
        self.first_has_arrived = 0
        self.has_arrived = [0] * world_size

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        sys.stdout.flush()

    def exposed_sync_initial_model(self, rank, init_period, tensor_received):
        # self.logging('receive rank %d, round_id 0' % rank)
        tensor_received = rpyc.classic.obtain(tensor_received)
        self.lock.acquire()
        if self.first_has_arrived == 0:
            self.first_has_arrived = 1
            self.flattened_params_length = len(tensor_received)
            self.averaged_tensor = torch.zeros(self.flattened_params_length)

            self.grad_consistency = torch.full([self.flattened_params_length], 1, dtype=torch.float32)
            self.min_grad_consistency = torch.full([self.flattened_params_length], 1, dtype=torch.float32)
            self.window_start_round = torch.zeros(self.flattened_params_length)
            self.ema_positive_grad = torch.zeros(self.flattened_params_length)
            self.ema_negative_grad = torch.zeros(self.flattened_params_length)
            self.param_sync_period = torch.full([self.flattened_params_length], self.initial_period, dtype=torch.int32)
            self.observe_window_size = torch.full([self.flattened_params_length], CAS_hyperparams_profile['observe_window_size'], dtype=torch.int32)
        self.lock.release()

        ''' averaging '''
        self.averaged_tensor += tensor_received/self.world_size

        self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.last_tensor_received = copy.deepcopy(self.averaged_tensor)
            ''' reset '''
            self.has_arrived = [0]*self.world_size
            self.first_has_arrived = 0
            
        self.lock.release()

        while self.has_arrived[rank] == 1:
            time.sleep(0.01)

        # self.logging('return rank %d, round_id 0' % rank)
        return self.averaged_tensor

    def exposed_aggregate(self, rank, round_id, tensor_received, temp_period):
        # self.logging('receive rank %d, round_id %d' % (rank, round_id))
        tensor_received = rpyc.classic.obtain(tensor_received)
        received_length = len(tensor_received)
        self.lock.acquire()
        if self.first_has_arrived == 0:
            self.mask = (self.param_sync_period == temp_period)
            self.averaged_tensor = torch.zeros(received_length)
            self.positive_grad = torch.zeros(received_length)
            self.negative_grad = torch.zeros(received_length)
            self.first_has_arrived = 1
            self.round_id = round_id
        self.lock.release()

        ''' averaging '''
        self.averaged_tensor += tensor_received/self.world_size

        ''' pooling '''
        grad_tensor = tensor_received - self.last_tensor_received[self.mask]
        self.positive_grad += torch.where(grad_tensor > 0, grad_tensor, 0)
        self.negative_grad -= torch.where(grad_tensor <= 0, grad_tensor, 0)

        self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.lock.release()
            ''' smoothing '''
            self.ema_positive_grad[self.mask] = self.eam_theta*self.ema_positive_grad[self.mask] + (1-self.eam_theta)*self.positive_grad
            self.ema_negative_grad[self.mask] = self.eam_theta*self.ema_negative_grad[self.mask] + (1-self.eam_theta)*self.negative_grad
        
            self.last_tensor_received[self.mask] = self.averaged_tensor

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.first_has_arrived = 0

        else:
            self.lock.release()
        
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        
        # self.logging('return rank %d, round_id %d' % (rank, round_id))
        return self.averaged_tensor
    
    def exposed_update_period(self, rank):
        ''' calculate grad consistency '''
        self.lock.acquire()
        self.has_arrived[rank] = 1
        if self.first_has_arrived == 0:
            self.first_has_arrived = 1
            self.lock.release()
            self.grad_consistency = abs(self.ema_positive_grad-self.ema_negative_grad) / (self.ema_positive_grad+self.ema_negative_grad)
            self.tune_mask = ((self.grad_consistency > self.min_grad_consistency) \
                                & (self.grad_consistency < 0.1) & (self.param_sync_period > 1) \
                                & (self.round_id > self.window_start_round + self.observe_window_size))
            self.temp_mask2 = (self.grad_consistency <= self.min_grad_consistency)

            self.param_sync_period[self.tune_mask] //= 2
            self.min_grad_consistency[self.tune_mask] = 1
            self.observe_window_size[self.tune_mask] *= 2
            self.ema_positive_grad[self.tune_mask] = 0
            self.ema_negative_grad[self.tune_mask] = 0
            self.window_start_round[self.tune_mask] = self.round_id

            self.window_start_round[self.temp_mask2] = self.round_id
            self.min_grad_consistency[self.temp_mask2] = self.grad_consistency[self.temp_mask2]

            index = self.tune_mask.nonzero()
            self.global_tune_index = torch.cat((index, self.tune_global_period()), 0)   

            while sum(self.has_arrived) != self.world_size:
                time.sleep(0.01)

            ''' reset '''
            self.has_arrived = [0]*self.world_size
            self.first_has_arrived = 0

        else:
            self.has_arrived[rank] = 1
            self.lock.release()

        while self.has_arrived[rank] == 1:
            time.sleep(0.01)

        return self.global_tune_index
    
    def tune_global_period(self):
        temp_period = self.initial_period
        proportion_list = []
        grad_consistency_list = []
        tune = torch.tensor([[0]])
        while temp_period >= 1:
            mask = (self.param_sync_period == temp_period)
            index = mask.nonzero()
            temp_len = len(index)
            if temp_len != 0:
                proportion = temp_len/self.flattened_params_length
                proportion_list.append(round(proportion*100,3))
                grad_consistency = self.grad_consistency[mask].sum() / temp_len
                if temp_period == self.global_period and proportion < 0.01: 
                    tune = torch.tensor([[1]])
            else:
                if temp_period == self.global_period:
                    tune = torch.tensor([[1]])
                else:
                    proportion_list.append(0)
                    grad_consistency = 0       
            grad_consistency_list.append(round(float(grad_consistency),4))
            temp_period //= 2
        self.logging("sync_period_distribution: "+str(proportion_list))
        self.logging("param_averaging_grad_consistency: "+str(grad_consistency_list))
        average_consistency = self.grad_consistency.sum() / self.flattened_params_length  
        self.logging("round_id: %d; average_consistency: %.4f" % (self.round_id, average_consistency))
        print()

        if tune == 1:
            mask = (self.param_sync_period == self.global_period)
            self.logging('tune global period')
            self.global_period //= 2
            self.param_sync_period[mask] //= 2
            self.observe_window_size[mask] *= 2
            self.ema_positive_grad[mask] = 0
            self.ema_negative_grad[mask] = 0
            self.min_grad_consistency[mask] = 1
            # self.window_start_round[mask] = self.round_id
        
        return tune

class Default_AggregationService(Service):
    def __init__(self, world_size):
        self.world_size = world_size
        self.round_id = 0
        self.flattened_params_length = 0
        self.lock = Lock()

        self.averaged_tensor = None
        self.has_arrived = [0] * world_size
        self.first_has_arrived = 0

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        sys.stdout.flush()

    def exposed_sync_initial_model(self, rank, tensor_received):
        # self.logging('receive rank %d, sync_initial_model' % (rank))
        tensor_received = rpyc.classic.obtain(tensor_received)
        print(tensor_received.device)
        self.lock.acquire()
        if self.first_has_arrived == 0:
            self.first_has_arrived = 1
            self.flattened_params_length = len(tensor_received)
            self.averaged_tensor = torch.zeros(self.flattened_params_length)
            print(self.averaged_tensor.device)
        self.lock.release()

        ''' averaging '''
        self.averaged_tensor += tensor_received/self.world_size

        self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            ''' reset '''
            self.has_arrived = [0]*self.world_size
            self.first_has_arrived = 0
            
        self.lock.release()

        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        
        return self.averaged_tensor


    def exposed_aggregate(self, rank, round_id, tensor_received):
        # self.logging('receive rank %d, round_id %d' % (rank, round_id))
        tensor_received = rpyc.classic.obtain(tensor_received)
        self.lock.acquire()
        if self.first_has_arrived == 0:
            self.first_has_arrived = 1
            self.averaged_tensor = torch.zeros(self.flattened_params_length)
        self.lock.release()
        self.round_id = max(self.round_id, round_id)
        self.averaged_tensor += tensor_received/self.world_size

        self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.first_has_arrived = 0
        self.lock.release()

        ''' otherwise, keep spining (blocked) '''
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)

        # self.logging('return rank %d, round_id %d' % (rank, round_id))
        return self.averaged_tensor


if __name__ == "__main__":

    ''' Parse arguments and create profile. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=20000)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')
    parser.add_argument('--interlayer_type', type=str, default='Defualt')
    parser.add_argument('--init_period', type=int, default=128)

    args = parser.parse_args()
    print('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks + '\n')
    sys.stdout.flush()

    if args.interlayer_type == 'CAS':
        service = CAS_AggregationService(args.world_size, args.init_period)
    else:
        service = Default_AggregationService(args.world_size)
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    server = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    server.start()
