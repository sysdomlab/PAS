import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time, random

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from threading import Lock,Thread

torch.set_printoptions(precision=4)
CUDA = torch.cuda.is_available()

CAS_hyperparams_profile = {
    'ema_theta' : 0.9,
    'observe_window_size' : 5,
    'tune_threshold' : 0.1,
}

reporting_fraction = 1

class CAS_AggregationService(Service):
    def __init__(self, world_size, init_period, test_process_port) -> None:
        super().__init__()
        self.flattened_params_length = None
        self.layer_len_list = []
        self.world_size = world_size
        self.receive_size = int(world_size*reporting_fraction)
        self.averaged_tensor_list = None
        self.eam_theta = CAS_hyperparams_profile['ema_theta']
        self.lock = Lock()
        self.round_id = 0
        self.initial_period = init_period
        self.global_period = init_period
        # self.last_global_proportion = 1
        # self.last_global_proportion_stable_round = 0
        self.min_averaged_grad_consistency = 1
        self.average_window_start_round = 0
        self.average_window_size = CAS_hyperparams_profile['observe_window_size']*2
        self.tune_threshold = CAS_hyperparams_profile['tune_threshold']


        self.layer_sync_period = []
        self.last_tensor_received = None
        self.ema_positive_grad = []
        self.ema_negative_grad = []
        self.sync_layer_id = None
        self.grad_consistency = []
        self.min_grad_consistency = []
        self.window_start_round = []
        self.global_tune_index = None
        self.observe_window_size = None
        self.layer_num = 0
        self.tune_tag = []
        
        self.first_has_arrived = 0
        self.has_arrived = [0] * self.world_size
        self.tensor_collected_list = [0] * self.receive_size
        self.reporting_index = None
        random.seed(0)

        ''' connect to the Test Process '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect('localhost', test_process_port, config=rpyc_config)
        self.connection._config['sync_request_timeout']=120
        print('- CAS Hyperparams Profile: ')
        pprint.pprint(CAS_hyperparams_profile)
        sys.stdout.flush()

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        sys.stdout.flush()

    def exposed_sync_initial_model(self, rank, tensor_list_received):
        # self.logging('receive rank %d, round_id 0' % rank)
        self.lock.acquire()
        if self.reporting_index == None:
            self.reporting_index = random.sample(range(self.world_size), self.receive_size)
        if rank in self.reporting_index:
            index = self.reporting_index.index(rank)
        else:
            index = -1
        self.lock.release()

        if index != -1:
            t1 = time.time()
            self.tensor_collected_list[index] = rpyc.classic.obtain(tensor_list_received)
            t2 = time.time()
            self.logging('rank %d, round_id 0, obtain time %.4f' % (rank, t2-t1))
            self.lock.acquire()
            if self.first_has_arrived == 0:
                self.first_has_arrived = 1
                for param in self.tensor_collected_list[index]:
                    temp_len = len(param.view(-1))
                    self.layer_len_list.append(temp_len)
                    self.grad_consistency.append(torch.full([temp_len], 1, dtype=torch.float32))
                    self.tune_tag.append(torch.zeros(temp_len))
                    self.min_grad_consistency.append(torch.full([temp_len], 1, dtype=torch.float32))
                    self.window_start_round.append(torch.zeros(temp_len))
                    self.ema_positive_grad.append(torch.zeros(temp_len))
                    self.ema_negative_grad.append(torch.zeros(temp_len))
                    self.layer_num += 1
                self.layer_sync_period = [self.initial_period] * self.layer_num
                self.observe_window_size = [CAS_hyperparams_profile['observe_window_size']] * self.layer_num
        else:
            self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.averaged_tensor_list = self.calculate_average()
            self.last_tensor_received = copy.deepcopy(self.averaged_tensor_list)

            ''' reset '''
            self.has_arrived = [0]*self.world_size
            self.tensor_collected_list = [0]*self.receive_size
            self.first_has_arrived = 0
            self.reporting_index = None
            self.logging('finishing round 0')
        self.lock.release()

        while self.has_arrived[rank] == 1:
            time.sleep(0.01)

        # self.logging('return rank %d, round_id 0' % rank)
        return self.averaged_tensor_list

    def exposed_aggregate(self, rank, round_id, tensor_list_received, temp_period):
        # self.logging('receive rank %d, round_id %d, temp_period %d' % (rank, round_id, temp_period))
        self.lock.acquire()
        if self.reporting_index == None:
            self.reporting_index = random.sample(range(self.world_size), self.receive_size)
        if rank in self.reporting_index:
            index = self.reporting_index.index(rank)
        else:
            index = -1
        self.lock.release()

        if index != -1:
            # t1 = time.time()
            self.tensor_collected_list[index] = rpyc.classic.obtain(tensor_list_received)
            # t2 = time.time()
            # self.logging('rank %d ontain_time %.4f' % (rank, t2-t1))
            self.lock.acquire()
            if self.first_has_arrived == 0:
                self.sync_layer_id = []
                for i in range(self.layer_num):
                    if self.layer_sync_period[i] == temp_period:
                        self.sync_layer_id.append(i)
                self.first_has_arrived = 1
                self.round_id = round_id
        else:
            self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.averaged_tensor_list = self.calculate_average()
            self.pooling_and_smoothing()
    
            for i in range(len(self.sync_layer_id)):
                self.last_tensor_received[self.sync_layer_id[i]] = self.averaged_tensor_list[i]

            if temp_period == self.global_period:
                ''' testing '''
                self.connection.root.test(self.last_tensor_received, round_id)

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_collected_list = [0] * self.receive_size
            self.first_has_arrived = 0
            # self.reporting_index = None
            # self.logging('finishing round %d' % round_id)
        self.lock.release()
        
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        
        # self.logging('return rank %d, round_id %d, temp_period %d' % (rank, round_id, temp_period))
        return self.averaged_tensor_list

    def exposed_aggregate_and_update(self, rank, round_id, tensor_list_received, temp_period):
        # self.logging('receive rank %d, round_id %d, temp_period %d' % (rank, round_id, temp_period))
        self.lock.acquire()
        if self.reporting_index == None:
            self.reporting_index = random.sample(range(self.world_size), self.receive_size)
        if rank in self.reporting_index:
            index = self.reporting_index.index(rank)
        else:
            index = -1
        self.lock.release()

        if index != -1:
            t1 = time.time()
            self.tensor_collected_list[index] = rpyc.classic.obtain(tensor_list_received)
            t2 = time.time()
            self.logging('rank %d ontain_time %.4f' % (rank, t2-t1))
            self.lock.acquire()
            if self.first_has_arrived == 0:
                self.sync_layer_id = []
                for i in range(self.layer_num):
                    if self.layer_sync_period[i] == temp_period:
                        self.sync_layer_id.append(i)
                self.first_has_arrived = 1
                self.round_id = round_id
        else:
            self.lock.acquire()

        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.logging('last receive rank %d, round_id %d, temp_period %d' % (rank, round_id, temp_period))
            self.averaged_tensor_list = self.calculate_average()
            self.pooling_and_smoothing()

            
            for i in range(len(self.sync_layer_id)):
                self.last_tensor_received[self.sync_layer_id[i]] = self.averaged_tensor_list[i]
            self.update_period()

            ''' testing '''
            self.connection.root.test(self.last_tensor_received, round_id)

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_collected_list = [0] *self.receive_size
            self.first_has_arrived = 0
            self.reporting_index = None
            self.logging('finishing round %d' % round_id)
        self.lock.release()
        
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        # self.logging('return rank %d, round_id %d, temp_period %d' % (rank, round_id, temp_period))
        return self.averaged_tensor_list
        
    def calculate_average(self):
        aggregated_tensor_list = []
        for tensor_id, tensor_content in enumerate(self.tensor_collected_list[0]):
            # calculate average layer by layer
            sum_tensor = torch.zeros(tensor_content.size())
            for rank in range(self.receive_size):
                sum_tensor += self.tensor_collected_list[rank][tensor_id]
            aggregated_tensor = sum_tensor / self.receive_size
            aggregated_tensor_list.append(aggregated_tensor)
        return aggregated_tensor_list

    def pooling_and_smoothing(self):
        ''' pooling '''
        for tensor_id, tensor_content in enumerate(self.tensor_collected_list[0]):
            positive_grad = torch.zeros(self.layer_len_list[self.sync_layer_id[tensor_id]])
            negative_grad = torch.zeros(self.layer_len_list[self.sync_layer_id[tensor_id]])
            for rank in range(self.receive_size):
                grad_tensor = (self.tensor_collected_list[rank][tensor_id] - self.last_tensor_received[self.sync_layer_id[tensor_id]]).view(-1)
                positive_grad += torch.where(grad_tensor > 0, grad_tensor, 0)
                negative_grad -= torch.where(grad_tensor <= 0, grad_tensor, 0)

            ''' smoothing '''
            self.ema_positive_grad[self.sync_layer_id[tensor_id]] = self.eam_theta*self.ema_positive_grad[self.sync_layer_id[tensor_id]] + (1-self.eam_theta)*positive_grad
            self.ema_negative_grad[self.sync_layer_id[tensor_id]] = self.eam_theta*self.ema_negative_grad[self.sync_layer_id[tensor_id]] + (1-self.eam_theta)*negative_grad
    

    def update_period(self):
        ''' calculate grad consistency '''
        layer_aver_consistency = []
        layer_stable_proportion = []
        tune_list = []
        for i in range(self.layer_num):
            self.grad_consistency[i] = abs(self.ema_positive_grad[i]-self.ema_negative_grad[i]) / (self.ema_positive_grad[i]+self.ema_negative_grad[i])
            mask1 = ((self.grad_consistency[i] > self.min_grad_consistency[i]) \
                                & (self.grad_consistency[i] < self.tune_threshold) \
                                & (self.round_id > self.window_start_round[i] + self.observe_window_size[i]))
            self.tune_tag[i][mask1]=1

            mask2 = ((self.grad_consistency[i] <= self.min_grad_consistency[i]) | (self.grad_consistency[i] > self.tune_threshold))
            self.window_start_round[i][mask2] = self.round_id
            self.min_grad_consistency[i][mask2] = self.grad_consistency[i][mask2]

            layer_aver_consistency.append(round(float(sum(self.grad_consistency[i])/self.layer_len_list[i]),4))
            layer_stable_proportion.append(round(float(sum(self.tune_tag[i])/self.layer_len_list[i]),4))
            if (layer_stable_proportion[i]>0.99 or self.layer_len_list[i] == sum(self.grad_consistency[i])+1) and self.layer_sync_period[i] > self.initial_period//16:
                tune_list.append(i)
                self.layer_sync_period[i] //= 2
                self.window_start_round[i].fill_(self.round_id)
                self.observe_window_size[i] *= 2
                self.min_grad_consistency[i].fill_(1)
                self.tune_tag[i].fill_(0)
                self.ema_positive_grad[i] = torch.zeros(self.layer_len_list[i])
                self.ema_negative_grad[i] = torch.zeros(self.layer_len_list[i])
        
        self.logging('layer_aver_consistency: '+str(layer_aver_consistency))
        self.logging('layer_stable_proportion: '+str(layer_stable_proportion))
        self.logging('layer_period: '+str(self.layer_sync_period))
        if self.global_period not in self.layer_sync_period:
            self.global_period //= 2
            self.logging('global period tune')
        self.averaged_tensor_list.append(tune_list)

    
    # def tune_global_period(self):
    #     temp_period = self.initial_period
    #     proportion_list = []
    #     grad_consistency_list = []
    #     tune = torch.tensor([[0]])
    #     while temp_period >= self.initial_period//16:
    #         mask = (self.param_sync_period == temp_period)
    #         index = mask.nonzero()
    #         temp_len = len(index)
    #         if temp_len != 0:
    #             proportion = temp_len/self.flattened_params_length
    #             proportion_list.append(round(proportion*100,3))
    #             grad_consistency = self.grad_consistency[mask].sum() / temp_len
    #             if temp_period == self.global_period:
    #                 # if proportion != self.last_global_proportion:
    #                 #     self.last_global_proportion = proportion
    #                 #     self.last_global_proportion_stable_round = 0
    #                 # else: self.last_global_proportion_stable_round += 1
    #                 if proportion < 0.01 : 
    #                     tune = torch.tensor([[1]])
    #         else:
    #             if temp_period == self.global_period:
    #                 tune = torch.tensor([[1]])
    #             else:
    #                 proportion_list.append(0)
    #                 grad_consistency = 0       
    #         grad_consistency_list.append(round(float(grad_consistency),4))
    #         temp_period //= 2
    #     self.logging("sync_period_distribution: "+str(proportion_list))
    #     self.logging("param_averaging_grad_consistency: "+str(grad_consistency_list))
    #     average_consistency = self.grad_consistency.sum() / self.flattened_params_length  
    #     sorted_value, sorted_index = torch.sort(self.grad_consistency)
    #     ten_part = int(self.flattened_params_length*0.1)
    #     some_grad_consistency_list = []
    #     for i in range(11):
    #         some_grad_consistency_list.append(round(float(sorted_value[ten_part*i]),4))
    #     self.logging("some_grad_consistency_list: "+str(some_grad_consistency_list))
    #     # if average_consistency < self.min_averaged_grad_consistency:
    #     #     self.min_averaged_grad_consistency = average_consistency
    #     #     self.average_window_start_round = self.round_id
    #     # if self.round_id > self.average_window_start_round + self.average_window_size and self.global_period > self.initial_period//16:
    #     #     tune = torch.tensor([[1]])
    #     self.logging("round_id: %d; average_consistency: %.4f" % (self.round_id, average_consistency))

    #     if tune == 1:
    #         mask = (self.param_sync_period == self.global_period)
    #         self.logging('tune global period')
    #         # self.average_window_start_round = self.round_id
    #         # self.min_averaged_grad_consistency = 1
    #         # self.average_window_size *= 2
    #         self.global_period //= 2
    #         self.param_sync_period[mask] //= 2
    #         self.observe_window_size[mask] *= 2
    #         self.ema_positive_grad[mask] = 0
    #         self.ema_negative_grad[mask] = 0
    #         self.min_grad_consistency[mask] = 1
    #         self.last_global_proportion = 1
    #         self.last_global_proportion_stable_round = 0
        
    #     return tune

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
    parser.add_argument('--test_process_port', type=int, default=30000)

    args = parser.parse_args()
    print('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks + '\n')
    sys.stdout.flush()

    random.seed(0)

    if args.interlayer_type == 'CAS':
        service = CAS_AggregationService(args.world_size, args.init_period, args.test_process_port)
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    server = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    server.start()