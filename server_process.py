import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from threading import Lock,Thread
from math import log2

from model_manager import Model_Manager
from dataset_manager import Dataset_Manager

torch.set_printoptions(precision=20)
CUDA = torch.cuda.is_available()

CAS_hyperparams_profile = {
    'ema_theta' : 0.9,
    'observe_window_size' : 5
}

test_lock = Lock()
model = None
testing_dataloader = None
cutoff_index_list, param_shape_list = [], []

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
        self.tensor_collected_list = [0] * world_size
        print('- CAS Hyperparams Profile: ')
        pprint.pprint(CAS_hyperparams_profile)
        sys.stdout.flush()

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        sys.stdout.flush()

    def exposed_sync_initial_model(self, rank, tensor_received):
        # self.logging('receive rank %d, round_id 0' % rank)
        self.tensor_collected_list[rank] = rpyc.classic.obtain(tensor_received)
        self.lock.acquire()
        if self.first_has_arrived == 0:
            self.first_has_arrived = 1
            self.flattened_params_length = len(self.tensor_collected_list[rank])

            self.grad_consistency = torch.full([self.flattened_params_length], 1, dtype=torch.float32)
            self.min_grad_consistency = torch.full([self.flattened_params_length], 1, dtype=torch.float32)
            self.window_start_round = torch.zeros(self.flattened_params_length)
            self.ema_positive_grad = torch.zeros(self.flattened_params_length)
            self.ema_negative_grad = torch.zeros(self.flattened_params_length)
            self.param_sync_period = torch.full([self.flattened_params_length], self.initial_period, dtype=torch.int32)
            self.observe_window_size = torch.full([self.flattened_params_length], CAS_hyperparams_profile['observe_window_size'], dtype=torch.int32)

        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.averaged_tensor = sum(self.tensor_collected_list)/self.world_size
            self.last_tensor_received = copy.deepcopy(self.averaged_tensor)
            ''' reset '''
            self.has_arrived = [0]*self.world_size
            self.tensor_collected_list = [0]*self.world_size
            self.first_has_arrived = 0
            self.logging('finishing round 0')
        self.lock.release()

        while self.has_arrived[rank] == 1:
            time.sleep(0.01)

        # self.logging('return rank %d, round_id 0' % rank)
        return self.averaged_tensor

    def exposed_aggregate(self, rank, round_id, tensor_received, temp_period):
        # self.logging('receive rank %d, round_id %d, temp_period %d' % (rank, round_id, temp_period))
        self.tensor_collected_list[rank] = rpyc.classic.obtain(tensor_received)
        self.lock.acquire()
        if self.first_has_arrived == 0:
            self.mask = (self.param_sync_period == temp_period)
            received_length = len(self.tensor_collected_list[rank])
            self.positive_grad = torch.zeros(received_length)
            self.negative_grad = torch.zeros(received_length)
            self.first_has_arrived = 1
            self.round_id = round_id

        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.lock.release()
            self.averaged_tensor = sum(self.tensor_collected_list)/self.world_size
            self.pooling_and_smoothing()
    
            self.last_tensor_received[self.mask] = self.averaged_tensor

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_collected_list = [0] * self.world_size
            self.first_has_arrived = 0
            self.logging('finishing round %d' % round_id)
        else:
            self.lock.release()
        
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        
        # self.logging('return rank %d, round_id %d, temp_period %d' % (rank, round_id, temp_period))
        return self.averaged_tensor

    def exposed_aggregate_and_update(self, rank, round_id, tensor_received, temp_period):
        # self.logging('receive rank %d, round_id %d, temp_period %d' % (rank, round_id, temp_period))
        self.tensor_collected_list[rank] = rpyc.classic.obtain(tensor_received)
        self.lock.acquire()
        if self.first_has_arrived == 0:
            self.mask = (self.param_sync_period == temp_period)
            received_length = len(self.tensor_collected_list[rank])
            self.positive_grad = torch.zeros(received_length)
            self.negative_grad = torch.zeros(received_length)
            self.first_has_arrived = 1
            self.round_id = round_id


        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.lock.release()
            self.averaged_tensor = sum(self.tensor_collected_list)/self.world_size
            self.pooling_and_smoothing()

            
            self.last_tensor_received[self.mask] = self.averaged_tensor
            self.update_period()

            ''' testing '''
            t1 = time.time()
            test_tensor = copy.deepcopy(self.last_tensor_received)
            testing_thread = Thread(target=test_thread, args=(round_id, test_tensor,))
            testing_thread.start()
            t2 = time.time()
            self.logging("start_testing_thread: %.4f" % (t2-t1))

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_collected_list = [0] *self.world_size
            self.first_has_arrived = 0
            self.logging('finishing round %d' % round_id)
        else:
            self.lock.release()
        
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        # self.logging('return rank %d, round_id %d, temp_period %d' % (rank, round_id, temp_period))
        return [self.averaged_tensor, self.global_tune_index]
        

    def pooling_and_smoothing(self):
        ''' pooling '''
        for i in range(self.world_size):
            grad_tensor = self.tensor_collected_list[i] - self.last_tensor_received[self.mask]
            self.positive_grad += torch.where(grad_tensor > 0, grad_tensor, 0)
            self.negative_grad -= torch.where(grad_tensor <= 0, grad_tensor, 0)

        ''' smoothing '''
        self.ema_positive_grad[self.mask] = self.eam_theta*self.ema_positive_grad[self.mask] + (1-self.eam_theta)*self.positive_grad
        self.ema_negative_grad[self.mask] = self.eam_theta*self.ema_negative_grad[self.mask] + (1-self.eam_theta)*self.negative_grad
    

    def update_period(self):
        ''' calculate grad consistency '''

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
        self.tensor_collected_list = [0] * world_size
        self.first_has_arrived = 0

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        sys.stdout.flush()

    def exposed_sync_initial_model(self, rank, tensor_received):
        # self.logging('receive rank %d, sync_initial_model' % (rank))
        self.tensor_collected_list[rank] = rpyc.classic.obtain(tensor_received)
        self.lock.acquire()
        if self.first_has_arrived == 0:
            self.first_has_arrived = 1
            self.flattened_params_length = len(tensor_received)

        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.averaged_tensor = sum(self.tensor_collected_list)/self.world_size
            ''' reset '''
            self.has_arrived = [0]*self.world_size
            self.tensor_collected_list = [0]*self.world_size
            self.first_has_arrived = 0
            self.logging('finishing round 0')
        self.lock.release()

        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        
        return self.averaged_tensor


    def exposed_aggregate(self, rank, round_id, tensor_received):
        # self.logging('receive rank %d, round_id %d' % (rank, round_id))
        self.tensor_collected_list[rank] = rpyc.classic.obtain(tensor_received)
        self.round_id = max(self.round_id, round_id)

        self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.averaged_tensor = sum(self.tensor_collected_list)/self.world_size

            ''' testing '''
            t1 = time.time()
            test_tensor = copy.deepcopy(self.averaged_tensor)
            testing_thread = Thread(target=test_thread, args=(round_id, test_tensor,))
            testing_thread.start()
            t2 = time.time()
            self.logging("start_testing_thread: %.4f" % (t2-t1))

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_collected_list = [0] * self.world_size
            self.first_has_arrived = 0

            self.logging('finishing round %d' % round_id)
        self.lock.release()


        ''' otherwise, keep spining (blocked) '''
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        # self.logging('return rank %d, round_id %d' % (rank, round_id))
        return self.averaged_tensor


def test_thread(round_id, tensor_to_restore):
        global model, testing_dataloader, cutoff_index_list, param_shape_list, test_lock

        test_lock.acquire()
        accuracy = 0
        positive_test_number = 0.0
        total_test_number = 0.0
        # loss = 0.0
        ''' restore param '''
        for i, param in enumerate(model.parameters()):
            param.data = tensor_to_restore[cutoff_index_list[i][0]:cutoff_index_list[i][1]].view(param_shape_list[i])

        t1 = time.time()
        with torch.no_grad():
            for step, (test_x, test_y) in enumerate(testing_dataloader):
                if CUDA:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                test_output = model(test_x)
                # loss += self.loss_func(test_output, test_y)
                # pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                # positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
                pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
                total_test_number += len(test_y)
        # loss /= step
        # print("loss: %.4f" % loss)
        accuracy = positive_test_number / total_test_number
        t2 = time.time()
        logging('test_time: %.4f' % (t2-t1))
        logging(' - test - round_id: %d; accuracy: %.4f;' % (round_id, accuracy))
        test_lock.release()
        
def logging(string):
    print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
    sys.stdout.flush()


if __name__ == "__main__":

    ''' Parse arguments and create profile. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=20000)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')
    parser.add_argument('--interlayer_type', type=str, default='Defualt')
    parser.add_argument('--init_period', type=int, default=128)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)

    args = parser.parse_args()
    print('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks + '\n')
    sys.stdout.flush()

    """ Local Training Profile """
    model_name, dataset_name = args.model, args.dataset
    local_profile = {
        'model_profile' : {
            'model_name' : model_name
        },
        'dataset_profile' : {
            'dataset_name' : dataset_name,
            'is_iid' : False,
            'world_size' : 20,
            'partition_rank' : 0,
            'batch_size' : 64,
            'dirichlet_alpha' : 1
        }
    }
    
    ''' Model Initialization'''
    model_manager = Model_Manager(local_profile['model_profile'])
    model = model_manager.load_model()
    # optimizer = model_manager.get_optimizer()
    # loss_func = model_manager.get_loss_func()

    ''' Initialization Index List '''
    s_index = 0
    for param in model.parameters():
        param_shape_list.append(param.data.shape)
        flatten_params = param.data.view(-1)
        e_index = s_index + len(flatten_params)
        cutoff_index_list.append([s_index, e_index])
        s_index = e_index


    ''' Dataset Initialization '''
    dataset_manager = Dataset_Manager(local_profile['dataset_profile'])
    testing_dataloader = dataset_manager.get_testing_dataloader()


    if args.interlayer_type == 'CAS':
        service = CAS_AggregationService(args.world_size, args.init_period)
    else:
        service = Default_AggregationService(args.world_size)
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    server = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    server.start()
