import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time, random, math

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from threading import Lock,Thread

torch.set_printoptions(precision=4)

CAS_hyperparams_profile = {
    'ema_theta' : 0.9,
    'observe_window_size' : 5,
    'tune_threshold' : 0.1,
}

Bert_CAS_hyperparams_profile = {
    'ema_theta' : 0.8,
    'observe_window_size' : 3,
    'tune_threshold' : 1.0,
}

APF_hyperparams = {
    'freezing_check_frequency' : 1, # unit for freezing_frequency is 'round'
    'ema_alpha' : 0.95,
    'stable_threshold' : 0.05,
    'enable_random_freezing' : False
}


class CAS_AggregationService(Service):
    def __init__(self, world_size, init_period, test_process_port, reporting_fraction) -> None:
        super().__init__()
        self.flattened_params_length = None
        self.world_size = world_size
        self.receive_size = int(world_size*reporting_fraction)
        self.averaged_tensor = None
        self.eam_theta = CAS_hyperparams_profile['ema_theta']
        self.lock = Lock()
        self.round_id = 0
        self.initial_period = init_period
        self.global_period = init_period
        self.last_global_proportion = 1
        self.last_global_proportion_stable_round = 0
        self.min_averaged_grad_consistency = 1
        self.average_window_start_round = 0
        self.tune_threshold = CAS_hyperparams_profile['tune_threshold']
        self.divisor = 2


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
        
        # add: tune when lower than a threshold
        self.low_threshold = 0.05
        self.low_start_round = None
        
        self.first_has_arrived = 0
        self.has_arrived = [0] * self.world_size
        self.tensor_collected_list = [0] * self.receive_size
        self.reporting_index = None
        random.seed(0)

        ''' connect to the Test Process '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect('localhost', test_process_port, config=rpyc_config)
        self.connection._config['sync_request_timeout']=1200
        print('- CAS Hyperparams Profile: ')
        pprint.pprint(CAS_hyperparams_profile)
        sys.stdout.flush()

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        sys.stdout.flush()

    def exposed_sync_initial_model(self, rank, tensor_received):
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
            self.tensor_collected_list[index] = rpyc.classic.obtain(tensor_received)
            t2 = time.time()
            # self.logging('rank %d, round_id 0, obtain time %.4f' % (rank, t2-t1))
            self.lock.acquire()
            if self.first_has_arrived == 0:
                self.first_has_arrived = 1
                self.flattened_params_length = len(self.tensor_collected_list[index])
                self.grad_consistency = torch.full([self.flattened_params_length], 1, dtype=torch.float32)
                self.min_grad_consistency = torch.full([self.flattened_params_length], 1, dtype=torch.float32)
                self.window_start_round = torch.zeros(self.flattened_params_length)
                self.ema_positive_grad = torch.zeros(self.flattened_params_length)
                self.ema_negative_grad = torch.zeros(self.flattened_params_length)
                self.param_sync_period = torch.full([self.flattened_params_length], self.initial_period, dtype=torch.int32)
                self.observe_window_size = torch.full([self.flattened_params_length], CAS_hyperparams_profile['observe_window_size']*2, dtype=torch.int32)
                self.low_start_round = torch.zeros(self.flattened_params_length)
        else:
            self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.logging('rank %d last arrived' % (rank))
            self.averaged_tensor = sum(self.tensor_collected_list)/self.receive_size
            self.last_tensor_received = copy.deepcopy(self.averaged_tensor)

            ''' testing '''
            self.connection.root.test(self.averaged_tensor, self.round_id)

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
        return self.averaged_tensor

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
            t1 = time.time()
            self.tensor_collected_list[index] = rpyc.classic.obtain(tensor_list_received)[0]
            t2 = time.time()
            # self.logging('rank %d ontain_time %.4f' % (rank, t2-t1))
            self.lock.acquire()
            if self.first_has_arrived == 0:
                self.mask = (self.param_sync_period == temp_period)
                received_length = len(self.tensor_collected_list[index])
                self.positive_grad = torch.zeros(received_length)
                self.negative_grad = torch.zeros(received_length)
                self.first_has_arrived = 1
                self.round_id = round_id
        else:
            self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.logging('rank %d last arrived' % (rank))
            self.averaged_tensor = sum(self.tensor_collected_list)/self.receive_size
            t1 = time.time()
            self.pooling_and_smoothing()
            t2 = time.time()
            self.logging('pooling and smoothing time: %.4f' % (t2-t1))
    
            self.last_tensor_received[self.mask] = self.averaged_tensor

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
        return [self.averaged_tensor]

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
            self.tensor_collected_list[index] = rpyc.classic.obtain(tensor_list_received)[0]
            t2 = time.time()
            # self.logging('rank %d ontain_time %.4f' % (rank, t2-t1))
            self.lock.acquire()
            if self.first_has_arrived == 0:
                self.mask = (self.param_sync_period == temp_period)
                received_length = len(self.tensor_collected_list[index])
                self.positive_grad = torch.zeros(received_length)
                self.negative_grad = torch.zeros(received_length)
                self.first_has_arrived = 1
                self.round_id = round_id
        else:
            self.lock.acquire()

        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.logging('last receive rank %d, round_id %d, temp_period %d' % (rank, round_id, temp_period))
            self.averaged_tensor = sum(self.tensor_collected_list)/self.receive_size
            t1 = time.time()
            self.pooling_and_smoothing()

            
            self.last_tensor_received[self.mask] = self.averaged_tensor
            self.update_period()
            t2 = time.time()
            self.logging('additional time: %.4f' % (t2-t1))

            ''' testing '''
            if self.round_id % (self.initial_period//self.global_period) == 0:
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
        return [self.averaged_tensor, self.global_tune_index]

    def pooling_and_smoothing(self):
        ''' pooling '''
        for i in range(self.receive_size):
            grad_tensor = self.tensor_collected_list[i] - self.last_tensor_received[self.mask]
            self.positive_grad += torch.where(grad_tensor > 0, grad_tensor, 0)
            self.negative_grad -= torch.where(grad_tensor <= 0, grad_tensor, 0)

        ''' smoothing '''
        self.ema_positive_grad[self.mask] = self.eam_theta*self.ema_positive_grad[self.mask] + (1-self.eam_theta)*self.positive_grad
        self.ema_negative_grad[self.mask] = self.eam_theta*self.ema_negative_grad[self.mask] + (1-self.eam_theta)*self.negative_grad
    

    def update_period(self):
        ''' calculate grad consistency '''

        self.grad_consistency = abs(self.ema_positive_grad-self.ema_negative_grad) / (self.ema_positive_grad+self.ema_negative_grad)
        zero_mask = ((self.ema_positive_grad+self.ema_negative_grad) == 0)
        self.grad_consistency[zero_mask] = 0
        
        self.tune_mask = ((self.grad_consistency > self.min_grad_consistency) \
                            & (self.grad_consistency < self.tune_threshold) & (self.param_sync_period > self.initial_period//16) \
                            & (self.round_id > self.window_start_round + self.observe_window_size))
        self.temp_mask2 = ((self.grad_consistency <= self.min_grad_consistency) | (self.grad_consistency > self.tune_threshold))
        
        # low_mask = (self.grad_consistency > self.tune_threshold)
        # self.low_start_round[low_mask] = self.round_id
        # low_tune_mask = ((self.round_id > self.low_start_round + self.observe_window_size) & 
        #                  (self.param_sync_period > self.initial_period//16))
        # self.tune_mask = self.tune_mask | low_tune_mask
        

        self.param_sync_period[self.tune_mask] = (self.param_sync_period[self.tune_mask]/self.divisor).type(torch.int32)
        self.min_grad_consistency[self.tune_mask] = 1
        self.observe_window_size[self.tune_mask] = (self.observe_window_size[self.tune_mask]*self.divisor).type(torch.int32)
        self.ema_positive_grad[self.tune_mask] = 0
        self.ema_negative_grad[self.tune_mask] = 0
        self.window_start_round[self.tune_mask] = self.round_id
        
        self.low_start_round[self.tune_mask] = self.round_id

        self.window_start_round[self.temp_mask2] = self.round_id
        self.min_grad_consistency[self.temp_mask2] = self.grad_consistency[self.temp_mask2]

        index = self.tune_mask.nonzero()
        self.global_tune_index = torch.cat((index, self.tune_global_period()), 0)   

    


    def tune_global_period(self):
        temp_period = self.initial_period
        proportion_list = []
        grad_consistency_list = []
        tune = torch.tensor([[0]])
        while temp_period >= self.initial_period//20:
            mask = (self.param_sync_period == temp_period)
            index = mask.nonzero()
            temp_len = len(index)
            if temp_len != 0:
                proportion = (temp_len/self.flattened_params_length)*100
                proportion_list.append(round(proportion,3))
                grad_consistency = self.grad_consistency[mask].sum() / temp_len
                if temp_period == self.global_period:
                    if int(proportion) != self.last_global_proportion:
                        self.last_global_proportion = int(proportion)
                        self.last_global_proportion_stable_round = 0
                    else: self.last_global_proportion_stable_round += 1
                    if proportion < 5 : 
                        tune = torch.tensor([[1]])
            else:
                if temp_period == self.global_period:
                    tune = torch.tensor([[1]])
                else:
                    proportion_list.append(0)
                    grad_consistency = 0       
            grad_consistency_list.append(round(float(grad_consistency),4))
            temp_period = int(temp_period/self.divisor)
        self.logging("sync_period_distribution: "+str(proportion_list))
        self.logging("param_averaging_grad_consistency: "+str(grad_consistency_list))
        average_consistency = self.grad_consistency.sum() / self.flattened_params_length 
        sorted_value, sorted_index = torch.sort(self.grad_consistency)
        ten_part = int(self.flattened_params_length*0.1)
        some_grad_consistency_list = []
        for i in range(10):
            some_grad_consistency_list.append(round(float(sorted_value[ten_part*i]),4))
        some_grad_consistency_list.append(round(float(sorted_value[-1]),4))
        self.logging("some_grad_consistency_list: "+str(some_grad_consistency_list))
        # if average_consistency < self.min_averaged_grad_consistency:
        #     self.min_averaged_grad_consistency = average_consistency
        #     self.average_window_start_round = self.round_id
        self.logging("round_id: %d; average_consistency: %.4f" % (self.round_id, average_consistency))

        if tune == 1 and self.global_period > self.initial_period//16:
            mask = (self.param_sync_period == self.global_period)
            self.logging('tune global period')
            # self.average_window_start_round = self.round_id
            # self.min_averaged_grad_consistency = 1
            self.global_period = int(self.global_period/self.divisor)
            self.param_sync_period[mask] = (self.param_sync_period[mask]/self.divisor).type(torch.int32)
            self.observe_window_size[mask] = (self.observe_window_size[mask]*self.divisor).type(torch.int32)
            self.ema_positive_grad[mask] = 0
            self.ema_negative_grad[mask] = 0
            self.min_grad_consistency[mask] = 1
            self.window_start_round[mask] = self.round_id
            self.last_global_proportion = 1
            self.last_global_proportion_stable_round = 0
            return tune
        else: 
            return torch.tensor([[0]])



class CASAPF_AggregationService(CAS_AggregationService):
    def __init__(self, world_size, init_period, test_process_port, reporting_fraction) -> None:
        super().__init__(world_size, init_period, test_process_port, reporting_fraction)

        # APF hyperparameters
        self.apf_ema_alpha = APF_hyperparams['ema_alpha']
        self.stable_threshold = APF_hyperparams['stable_threshold'] 
        self.tighter_stable_criterion_threshold = 0.8
        self.enable_random_freezing = APF_hyperparams['enable_random_freezing']

        # APF variable
        self.active_mask = None
        self.grad_ema = None
        self.abs_grad_ema = None
        self.freezing_lengths = None
        self.fc_round_ids_to_unfreeze_params = None

        print('- APF Hyperparams Profile: ')
        pprint.pprint(APF_hyperparams)
        sys.stdout.flush()

    def exposed_sync_initial_model(self, rank, tensor_received):
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
            self.tensor_collected_list[index] = rpyc.classic.obtain(tensor_received)
            # self.logging('rank %d, round_id 0, obtain time %.4f' % (rank, t2-t1))
            self.lock.acquire()
            if self.first_has_arrived == 0:
                self.first_has_arrived = 1
                self.flattened_params_length = len(self.tensor_collected_list[index])
                self.grad_consistency = torch.full([self.flattened_params_length], 1, dtype=torch.float32)
                self.min_grad_consistency = torch.full([self.flattened_params_length], 1, dtype=torch.float32)
                self.window_start_round = torch.zeros(self.flattened_params_length)
                self.ema_positive_grad = torch.zeros(self.flattened_params_length)
                self.ema_negative_grad = torch.zeros(self.flattened_params_length)
                self.param_sync_period = torch.full([self.flattened_params_length], self.initial_period, dtype=torch.int32)
                self.observe_window_size = torch.full([self.flattened_params_length], CAS_hyperparams_profile['observe_window_size']*2, dtype=torch.int32)
                self.low_start_round = torch.zeros(self.flattened_params_length)
    

                self.active_mask = torch.ones(self.flattened_params_length).bool()
                self.grad_ema = torch.zeros(self.flattened_params_length, dtype=torch.float32)
                self.abs_grad_ema = torch.zeros(self.flattened_params_length, dtype=torch.float32)
                self.freezing_lengths = torch.zeros(self.flattened_params_length, dtype=torch.int32)
                self.fc_round_ids_to_unfreeze_params = torch.zeros(self.flattened_params_length, dtype=torch.int32)

        else:
            self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.logging('rank %d last arrived' % (rank))
            self.averaged_tensor = sum(self.tensor_collected_list)/self.receive_size
            self.last_tensor_received = copy.deepcopy(self.averaged_tensor)

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
        return self.averaged_tensor

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
            t1 = time.time()
            self.tensor_collected_list[index] = rpyc.classic.obtain(tensor_list_received)[0]
            t2 = time.time()
            # self.logging('rank %d ontain_time %.4f' % (rank, t2-t1))
            self.lock.acquire()
            if self.first_has_arrived == 0:
                self.mask = (self.param_sync_period == temp_period) & self.active_mask
                received_length = len(self.tensor_collected_list[index])
                self.positive_grad = torch.zeros(received_length)
                self.negative_grad = torch.zeros(received_length)
                self.first_has_arrived = 1
                self.round_id = round_id
        else:
            self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.logging('rank %d last arrived' % (rank))
            self.averaged_tensor = sum(self.tensor_collected_list)/self.receive_size
            t1 = time.time()
            self.pooling_and_smoothing()
            t2 = time.time()
            self.logging('pooling and smoothing time: %.4f' % (t2-t1))
    
            self.last_tensor_received[self.mask] = self.averaged_tensor

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
        return [self.averaged_tensor]

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
            self.tensor_collected_list[index] = rpyc.classic.obtain(tensor_list_received)[0]
            t2 = time.time()
            # self.logging('rank %d ontain_time %.4f' % (rank, t2-t1))
            self.lock.acquire()
            if self.first_has_arrived == 0:
                self.mask = (self.param_sync_period == temp_period) & self.active_mask
                received_length = len(self.tensor_collected_list[index])
                self.positive_grad = torch.zeros(received_length)
                self.negative_grad = torch.zeros(received_length)
                self.first_has_arrived = 1
                self.round_id = round_id
        else:
            self.lock.acquire()

        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.logging('last receive rank %d, round_id %d, temp_period %d' % (rank, round_id, temp_period))
            self.averaged_tensor = sum(self.tensor_collected_list)/self.receive_size
            t1 = time.time()
            self.pooling_and_smoothing()
            self.last_tensor_received[self.mask] = self.averaged_tensor

            self.update_period()
            self.refresh_freezing_status()

            t2 = time.time()
            self.logging('additional time: %.4f' % (t2-t1))

            ''' testing '''
            if self.round_id % (self.initial_period//self.global_period) == 0:
                self.connection.root.test(self.last_tensor_received, round_id)

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_collected_list = [0] * self.receive_size
            self.first_has_arrived = 0
            self.reporting_index = None
            self.logging('finishing round %d' % round_id)
        self.lock.release()
        
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        # self.logging('return rank %d, round_id %d, temp_period %d' % (rank, round_id, temp_period))
        return [self.averaged_tensor, self.global_tune_index]

    def update_param_freezing_periods_in_TCP_manner(self): 
        """ update the active mask of each parameter based on global gradient stability. """
        ''' Calculate the effective stepping rate of each parameter (in tensor mode for fast processing). '''
        effective_stepping_rate = torch.abs(self.grad_ema[self.active_mask]) / self.abs_grad_ema[self.active_mask]
        self.logging("effective_stepping_rate: {:.4f}".format(float(torch.mean(effective_stepping_rate))))

        ''' Update the freezing period length and also the should-be-unfrozen frequency_check_round_id based on effective stepping rate of each param. '''
        self.freezing_lengths[self.active_mask] = torch.where(effective_stepping_rate < self.stable_threshold, 
                                                               self.freezing_lengths[self.active_mask]+1, 
                                                               self.freezing_lengths[self.active_mask]//2)
        self.fc_round_ids_to_unfreeze_params[self.active_mask] = self.round_id + self.freezing_lengths[self.active_mask]

    def refresh_freezing_status(self):
        """ Called after each global synchronization is conducted. Update the freezing status and related statistics."""
        ''' Update the freezing period in a TCP manner and then update the freezing mask accordingly '''
        self.update_param_freezing_periods_in_TCP_manner()
        self.active_mask = (self.fc_round_ids_to_unfreeze_params == self.round_id)
        ''' Record the stable-ratio statistics and adaptively tune stability threshold if necessary '''
        self.freezing_ratio = 1 - self.active_mask.sum().float() / self.flattened_params_length
        self.logging('current freezing ratio: %.4f' % self.freezing_ratio)

    def pooling_and_smoothing(self):
        ''' pooling '''
        for i in range(self.receive_size):
            grad_tensor = self.tensor_collected_list[i] - self.last_tensor_received[self.mask]
            self.positive_grad += torch.where(grad_tensor > 0, grad_tensor, 0)
            self.negative_grad -= torch.where(grad_tensor <= 0, grad_tensor, 0)
        
        ''' apf '''
        avg_grad= self.averaged_tensor - self.last_tensor_received[self.mask]
        self.grad_ema[self.mask] = self.grad_ema[self.mask] * self.apf_ema_alpha + avg_grad* (1.0 - self.apf_ema_alpha) 
        self.abs_grad_ema[self.mask] = self.abs_grad_ema[self.mask] * self.apf_ema_alpha + torch.abs(avg_grad) * (1.0 - self.apf_ema_alpha)

        ''' smoothing '''
        self.ema_positive_grad[self.mask] = self.eam_theta*self.ema_positive_grad[self.mask] + (1-self.eam_theta)*self.positive_grad
        self.ema_negative_grad[self.mask] = self.eam_theta*self.ema_negative_grad[self.mask] + (1-self.eam_theta)*self.negative_grad
    

    def update_period(self):
        ''' calculate grad consistency '''

        self.grad_consistency[self.active_mask] = (abs(self.ema_positive_grad[self.active_mask]-self.ema_negative_grad[self.active_mask]) / 
                                                   (self.ema_positive_grad[self.active_mask]+self.ema_negative_grad[self.active_mask]))
        zero_mask = ((self.ema_positive_grad[self.active_mask]+self.ema_negative_grad[self.active_mask]) == 0)
        self.grad_consistency[self.active_mask][zero_mask] = 0
        
        self.tune_mask = ((self.grad_consistency > self.min_grad_consistency) \
                            & (self.grad_consistency < self.tune_threshold) & (self.param_sync_period > self.initial_period//16) \
                            & (self.round_id > self.window_start_round + self.observe_window_size))
        self.temp_mask2 = ((self.grad_consistency <= self.min_grad_consistency) | (self.grad_consistency > self.tune_threshold))
        
        low_mask = (self.grad_consistency > 0.1)
        self.low_start_round[low_mask] = self.round_id
        low_tune_mask = ((self.round_id > self.low_start_round + self.observe_window_size) & 
                         (self.param_sync_period > self.initial_period//16))
        self.tune_mask = (self.tune_mask | low_tune_mask) & self.active_mask
        

        self.param_sync_period[self.tune_mask] = (self.param_sync_period[self.tune_mask]/self.divisor).type(torch.int32)
        self.min_grad_consistency[self.tune_mask] = 1
        self.observe_window_size[self.tune_mask] = (self.observe_window_size[self.tune_mask]*self.divisor).type(torch.int32)
        self.ema_positive_grad[self.tune_mask] = 0
        self.ema_negative_grad[self.tune_mask] = 0
        self.window_start_round[self.tune_mask] = self.round_id
        
        self.low_start_round[self.tune_mask] = self.round_id

        self.window_start_round[self.temp_mask2] = self.round_id
        self.min_grad_consistency[self.temp_mask2] = self.grad_consistency[self.temp_mask2]

        index = self.tune_mask.nonzero()
        self.global_tune_index = torch.cat((index, self.tune_global_period()), 0)   

    
    def tune_global_period(self):
        temp_period = self.initial_period
        proportion_list = []
        grad_consistency_list = []
        tune = torch.tensor([[0]])
        while temp_period >= self.initial_period//20:
            mask = (self.param_sync_period == temp_period)
            index = mask.nonzero()
            temp_len = len(index)
            if temp_len != 0:
                proportion = (temp_len/self.flattened_params_length)*100
                proportion_list.append(round(proportion,3))
                grad_consistency = self.grad_consistency[mask].sum() / temp_len
                if temp_period == self.global_period:
                    if int(proportion) != self.last_global_proportion:
                        self.last_global_proportion = int(proportion)
                        self.last_global_proportion_stable_round = 0
                    else: self.last_global_proportion_stable_round += 1
                    if proportion < 5 : 
                        tune = torch.tensor([[1]])
            else:
                if temp_period == self.global_period:
                    tune = torch.tensor([[1]])
                else:
                    proportion_list.append(0)
                    grad_consistency = 0       
            grad_consistency_list.append(round(float(grad_consistency),4))
            temp_period = int(temp_period/self.divisor)
        self.logging("sync_period_distribution: "+str(proportion_list))
        self.logging("param_averaging_grad_consistency: "+str(grad_consistency_list))
        average_consistency = self.grad_consistency.sum() / self.flattened_params_length  
        average_activate_consistency = self.grad_consistency[self.active_mask].mean()
        sorted_value, sorted_index = torch.sort(self.grad_consistency)
        ten_part = int(self.flattened_params_length*0.1)
        some_grad_consistency_list = []
        for i in range(11):
            some_grad_consistency_list.append(round(float(sorted_value[ten_part*i]),4))
        self.logging("some_grad_consistency_list: "+str(some_grad_consistency_list))
        # if average_consistency < self.min_averaged_grad_consistency:
        #     self.min_averaged_grad_consistency = average_consistency
        #     self.average_window_start_round = self.round_id
        self.logging("round_id: %d; average_consistency: %.4f, average_active_consistency: %.4f" % (self.round_id, average_consistency, average_activate_consistency))

        if tune == 1 and self.global_period > self.initial_period//16:
            mask = (self.param_sync_period == self.global_period)
            self.logging('tune global period')
            # self.average_window_start_round = self.round_id
            # self.min_averaged_grad_consistency = 1
            self.global_period = int(self.global_period/self.divisor)
            self.param_sync_period[mask] = (self.param_sync_period[mask]/self.divisor).type(torch.int32)
            self.observe_window_size[mask] = (self.observe_window_size[mask]*self.divisor).type(torch.int32)
            self.ema_positive_grad[mask] = 0
            self.ema_negative_grad[mask] = 0
            self.min_grad_consistency[mask] = 1
            self.window_start_round[mask] = self.round_id
            self.last_global_proportion = 1
            self.last_global_proportion_stable_round = 0
            return tune
        else: 
            return torch.tensor([[0]])
    