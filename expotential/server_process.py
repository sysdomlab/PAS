import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time, random, math

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

GIFT_hyperparams_profile = {
    'ema_theta' : 0.9,
    'observe_window_size' : 5,
    'tune_threshold' : 0.1,
}



reporting_fraction = 0.4

class CAS_AggregationService(Service):
    def __init__(self, world_size, init_period, test_process_port) -> None:
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
        self.average_window_size = 20


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
        self.tune_mask = ((self.grad_consistency > self.min_grad_consistency) \
                            & (self.grad_consistency < self.tune_threshold) & (self.param_sync_period > self.initial_period//16) \
                            & (self.round_id > self.window_start_round + self.observe_window_size))
        self.temp_mask2 = ((self.grad_consistency <= self.min_grad_consistency) | (self.grad_consistency > self.tune_threshold))

        self.param_sync_period[self.tune_mask] = (self.param_sync_period[self.tune_mask]/self.divisor).type(torch.int32)
        self.min_grad_consistency[self.tune_mask] = 1
        self.observe_window_size[self.tune_mask] = (self.observe_window_size[self.tune_mask]*self.divisor).type(torch.int32)
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
                    if self.last_global_proportion_stable_round >= self.average_window_size:
                        tune = torch.tensor([[1]])
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
        for i in range(11):
            some_grad_consistency_list.append(round(float(sorted_value[ten_part*i]),4))
        self.logging("some_grad_consistency_list: "+str(some_grad_consistency_list))
        # if average_consistency < self.min_averaged_grad_consistency:
        #     self.min_averaged_grad_consistency = average_consistency
        #     self.average_window_start_round = self.round_id
        # if self.round_id > self.average_window_start_round + self.average_window_size  and self.global_period > 1:
        #     tune = torch.tensor([[1]])
        self.logging("round_id: %d; average_consistency: %.4f" % (self.round_id, average_consistency))

        if tune == 1 and self.global_period > self.initial_period//16:
            mask = (self.param_sync_period == self.global_period)
            self.logging('tune global period')
            # self.average_window_start_round = self.round_id
            # self.min_averaged_grad_consistency = 1
            self.average_window_size += int(20*self.divisor//2)
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


class Default_AggregationService(Service):
    def __init__(self, world_size, init_period, test_process_port):
        self.world_size = world_size
        self.init_period = init_period
        self.receive_size = int(world_size*reporting_fraction)
        self.round_id = 0
        self.flattened_params_length = 0
        self.lock = Lock()

        self.averaged_tensor = None
        self.has_arrived = [0] * world_size
        self.tensor_collected_list = [0] * self.receive_size
        self.reporting_index = None
        self.first_has_arrived = 0

        ''' connect to the Test Process '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect('localhost', test_process_port, config=rpyc_config)
        self.connection._config['sync_request_timeout']=120
        # self.test = rpyc.async_(self.connection.root.test)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        sys.stdout.flush()

    def exposed_sync_initial_model(self, rank, tensor_received):
        # self.logging('receive rank %d, sync_initial_model' % (rank))
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
            self.lock.acquire()
            if self.first_has_arrived == 0:
                self.first_has_arrived = 1
                self.flattened_params_length = len(self.tensor_collected_list[index])
        else:
            self.lock.acquire()

        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.logging('rank %d last arrived' % (rank))          
            self.averaged_tensor = sum(self.tensor_collected_list)/self.receive_size
            ''' reset '''
            self.has_arrived = [0]*self.world_size
            self.tensor_collected_list = [0]*self.receive_size
            self.first_has_arrived = 0
            self.reporting_index = None
            self.logging('finishing round 0')
        self.lock.release()

        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        
        return self.averaged_tensor


    def exposed_aggregate(self, rank, round_id, tensor_received):
        # self.logging('receive rank %d, round_id %d' % (rank, round_id))
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
            # self.logging('rank %d, obtain time %.4f' % (rank, t2-t1))
            self.round_id = round_id

        self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.logging('last receive rank %d, round_id %d' % (rank, round_id))
            self.averaged_tensor = sum(self.tensor_collected_list)/self.receive_size

            # if self.init_period >= 100 or round_id % (100//self.init_period) == 0:
            ''' testing '''
            self.connection.root.test(self.averaged_tensor, round_id)

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_collected_list = [0] * self.receive_size
            self.first_has_arrived = 0
            self.reporting_index = None

            self.logging('finishing round %d' % round_id)
        self.lock.release()


        ''' otherwise, keep spining (blocked) '''
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        # self.logging('return rank %d, round_id %d' % (rank, round_id))
        return self.averaged_tensor


class GIFT_AggregationService(Service):
    def __init__(self, world_size, init_period, test_process_port):
        self.world_size = world_size
        self.init_period = init_period
        self.current_period = init_period
        self.receive_size = int(world_size*reporting_fraction)
        self.round_id = 0
        self.flattened_params_length = 0
        self.lock = Lock()
        self.ema_theta = GIFT_hyperparams_profile['ema_theta']
        self.observe_window_size = GIFT_hyperparams_profile['observe_window_size']
        self.min_consistency = 1
        # self.window_start_round = 0
        self.ema_positive_grad = None
        self.ema_negative_grad = None
        self.last_tensor_received = None
        self.last_tune_round = 0
        self.min_grad_consistency = None
        self.window_start_round = None
        self.tune_tag = None
        self.tune=0

        self.averaged_tensor = None
        self.has_arrived = [0] * world_size
        self.tensor_collected_list = [0] * self.receive_size
        self.reporting_index = None
        self.first_has_arrived = 0

        ''' connect to the Test Process '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect('localhost', test_process_port, config=rpyc_config)
        self.connection._config['sync_request_timeout']=120
        # self.test = rpyc.async_(self.connection.root.test)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        sys.stdout.flush()

    def exposed_sync_initial_model(self, rank, tensor_received):
        # self.logging('receive rank %d, sync_initial_model' % (rank))
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
            self.lock.acquire()
            if self.first_has_arrived == 0:
                self.first_has_arrived = 1
                self.flattened_params_length = len(self.tensor_collected_list[index])
                self.ema_positive_grad = torch.zeros(self.flattened_params_length)
                self.ema_negative_grad = torch.zeros(self.flattened_params_length)
                self.min_grad_consistency = torch.full([self.flattened_params_length], 1, dtype=torch.float32)
                self.window_start_round = torch.zeros(self.flattened_params_length)
                self.tune_tag = torch.zeros(self.flattened_params_length)

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
        
        return self.averaged_tensor


    def exposed_aggregate(self, rank, round_id, tensor_received):
        # self.logging('receive rank %d, round_id %d' % (rank, round_id))
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
            # self.logging('rank %d, obtain time %.4f' % (rank, t2-t1))
            self.round_id = round_id

        self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.logging('last receive rank %d, round_id %d' % (rank, round_id))
            self.averaged_tensor = sum(self.tensor_collected_list)/self.receive_size
            self.pooling_and_smoothing()
            self.last_tensor_received = self.averaged_tensor

            self.tune = 0
            if (round_id-self.last_tune_round) % (self.init_period//self.current_period) == 0:
                self.tune = self.update_period()
                self.last_tune_round = self.round_id
            ''' testing '''
            self.connection.root.test(self.averaged_tensor, round_id)

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_collected_list = [0] * self.receive_size
            self.first_has_arrived = 0
            self.reporting_index = None

            self.logging('finishing round %d' % round_id)
        self.lock.release()


        ''' otherwise, keep spining (blocked) '''
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        # self.logging('return rank %d, round_id %d' % (rank, round_id))
        return [self.averaged_tensor, self.tune]

    def pooling_and_smoothing(self):
        positive_grad = torch.zeros(self.flattened_params_length)
        negative_grad = torch.zeros(self.flattened_params_length)
        
        ''' pooling '''
        for i in range(self.receive_size):
            grad_tensor = self.tensor_collected_list[i] - self.last_tensor_received
            positive_grad += torch.where(grad_tensor > 0, grad_tensor, 0)
            negative_grad -= torch.where(grad_tensor <= 0, grad_tensor, 0)

        ''' smoothing '''
        self.ema_positive_grad = self.ema_theta*self.ema_positive_grad + (1-self.ema_theta)*positive_grad
        self.ema_negative_grad = self.ema_theta*self.ema_negative_grad + (1-self.ema_theta)*negative_grad

    def update_period(self):
        grad_consistency = abs(self.ema_positive_grad-self.ema_negative_grad) / (self.ema_positive_grad+self.ema_negative_grad)

        mask2 = ((grad_consistency > self.min_grad_consistency) \
                 & (grad_consistency<0.1) & (self.round_id > self.window_start_round + self.observe_window_size))
        self.tune_tag[mask2] = 1


        mask1 = ((grad_consistency <= self.min_grad_consistency) | (grad_consistency > 0.1))
        self.window_start_round[mask1] = self.round_id
        self.min_grad_consistency[mask1] = grad_consistency[mask1]

        average_consistency = grad_consistency.sum()/self.flattened_params_length
        self.logging("averaging gradient consistency: %.4f, stable proportion: %.4f" % (average_consistency, self.tune_tag.sum()/self.flattened_params_length))
        # if (grad_consistency < self.min_consistency):
        #     self.min_consistency = grad_consistency
        #     self.window_start_round = self.round_id
        # else:
        if self.tune_tag.sum()>0.99*self.flattened_params_length and self.current_period > self.init_period//16:
            self.current_period //= 2
            self.window_start_round.fill_(self.round_id)
            self.observe_window_size *= 2
            self.min_grad_consistency.fill_(1)
            self.tune_tag.fill_(0)
            self.ema_positive_grad = torch.zeros(self.flattened_params_length)
            self.ema_negative_grad = torch.zeros(self.flattened_params_length)
            self.logging('sync period tune to %d' % self.current_period)
            return 1
        return 0


class AdaComm_AggregationService(Default_AggregationService):
    def __init__(self, world_size, init_period, test_process_port):
        super().__init__(world_size, init_period, test_process_port)
        
        # adacomm hyper param
        self.communication_computation_ratio = 240
        self.T = 6000
        
        # adacomm variant
        self.current_period = init_period
        self.round_loss_list = [0] * self.receive_size
        self.loss_in_observation_window = []
        self.init_loss = -1
        self.time = 0
        self.next_check_time = 0
        

    def exposed_aggregate(self, rank, round_id, tensor_received, loss):
        # self.logging('receive rank %d, round_id %d' % (rank, round_id))
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
            self.round_loss_list[index] = rpyc.classic.obtain(loss)
            t2 = time.time()
            # self.logging('rank %d, obtain time %.4f' % (rank, t2-t1))
            self.round_id = round_id

        self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.logging('last receive rank %d, round_id %d' % (rank, round_id))
            self.averaged_tensor = sum(self.tensor_collected_list)/self.receive_size
            self.update_period()

            # if self.init_period >= 100 or round_id % (100//self.init_period) == 0:
            ''' testing '''
            self.connection.root.test(self.averaged_tensor, round_id)

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_collected_list = [0] * self.receive_size
            self.round_loss_list = [0] * self.receive_size
            self.first_has_arrived = 0
            self.reporting_index = None

            self.logging('finishing round %d' % round_id)
        self.lock.release()


        ''' otherwise, keep spining (blocked) '''
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        # self.logging('return rank %d, round_id %d' % (rank, round_id))
        return self.averaged_tensor, self.current_period
    
    def update_period(self):
        self.loss_in_observation_window.append(sum(self.round_loss_list)/len(self.round_loss_list))
        self.time = self.time + self.current_period + self.communication_computation_ratio
        
        if self.time > self.next_check_time:
            mean_loss = sum(self.loss_in_observation_window) / len(self.loss_in_observation_window)
            if self.init_loss == -1:
                self.init_loss = mean_loss
                self.logging(f"init loss set to: {mean_loss:.2f}")
                self.loss_in_observation_window = []
                return
                
            self.current_period = min(int(math.sqrt((mean_loss/self.init_loss)) * self.init_period) + 1, self.init_period)
            self.logging(f"Current loss: {mean_loss:.4f}, init loss: {self.init_loss:.4f}, current period: {self.current_period}")
            self.logging('Revise sync period by loss!')
            self.next_check_time += self.T 
            self.loss_in_observation_window = []


class FedLAMA_AggregationService(Service):
    def __init__(self, world_size, init_period, test_process_port):
        self.world_size = world_size
        self.init_period = init_period
        self.receive_size = int(world_size*reporting_fraction)
        self.round_id = 0
        self.flattened_params_length = 0
        self.lock = Lock()

        self.global_tensor_list = None
        self.has_arrived = [0] * world_size
        self.tensor_collected_list = [0] * self.receive_size
        self.layer_list = [0] * self.receive_size
        self.reporting_index = None
        self.first_has_arrived = 0
        self.tensor_list_to_transmit = []
        
        # TODO: For FedLAMA
        self.phi = 2
        self.layer_num = 0
        self.sync_period_l = []
        self.next_sync_iter_l = []

        self.d_l = []
        self.delta_l = []
        self.lambda_l = []
        self.dim_l = []

        ''' connect to the Test Process '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect('localhost', test_process_port, config=rpyc_config)
        self.connection._config['sync_request_timeout']=120
        # self.test = rpyc.async_(self.connection.root.test)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        sys.stdout.flush()

    def exposed_sync_initial_model(self, rank, tensor_list_received):
        # self.logging('receive rank %d, sync_initial_model' % (rank))
        self.lock.acquire()
        if self.reporting_index == None:
            self.reporting_index = random.sample(range(self.world_size), self.receive_size)
        if rank in self.reporting_index:
            index = self.reporting_index.index(rank)
        else:
            index = -1
        self.lock.release()

        if index != -1:
            self.tensor_collected_list[index] = rpyc.classic.obtain(tensor_list_received)
            self.lock.acquire()
            if self.first_has_arrived == 0:
                self.first_has_arrived = 1
                self.layer_num = len(tensor_list_received)
                self.d_l = [0.] * self.layer_num
                self.delta_l = [0.] * self.layer_num
                self.lambda_l = [0.] * self.layer_num
                self.sync_period_l = [self.init_period] * self.layer_num
                self.next_sync_iter_l = [self.init_period] * self.layer_num
                for param in enumerate(tensor_list_received):
                    self.dim_l.append(len(param))
                self.global_tensor_list = copy.deepcopy(tensor_list_received)
        else:
            self.lock.acquire()

        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.logging('rank %d last arrived' % (rank))     
            for i in range(self.layer_num):
                self.global_tensor_list[i] = sum([param[i] for param in self.tensor_collected_list])/self.receive_size 
            ''' reset '''
            self.has_arrived = [0]*self.world_size
            self.tensor_collected_list = [0]*self.receive_size
            self.first_has_arrived = 0
            self.reporting_index = None
            self.logging('finishing round 0')
        self.lock.release()

        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        
        return self.global_tensor_list


    def exposed_aggregate(self, rank, iter_id, tensor_list_received, layer_list_received):
        # self.logging('receive rank %d, round_id %d' % (rank, round_id))
        self.lock.acquire()
        if self.reporting_index == None:
            self.reporting_index = random.sample(range(self.world_size), self.receive_size)
        if rank in self.reporting_index:
            index = self.reporting_index.index(rank)
        else:
            index = -1
        self.lock.release()

        if index != -1:
            self.tensor_collected_list[index] = rpyc.classic.obtain(tensor_list_received)
            self.layer_list[index] = rpyc.classic.obtain(layer_list_received)

        self.lock.acquire()
        self.has_arrived[rank] = 1
        if sum(self.has_arrived) == self.world_size:
            self.round_id += 1
            self.logging('last receive rank %d, round_id %d' % (rank, self.round_id))
            
            param_num_transmit = 0
            for layer_id in self.layer_list[0]:
                param_num_transmit += self.dim_l[layer_id]
            sparsification_fraction = param_num_transmit/sum(self.dim_l)
            self.logging(f'round_id {self.round_id}, transmit ratio: {sparsification_fraction*100:.2f}%, layers: {self.layer_list[0]}')
            
            ''' aggregate '''
            self.tensor_list_to_transmit = []
            for index, layer_id in enumerate(self.layer_list[0]):
                self.global_tensor_list[layer_id] = sum(param[index] for param in self.tensor_collected_list)/self.receive_size
                self.tensor_list_to_transmit.append(self.global_tensor_list[layer_id])
                
            self.update_d_l(self.layer_list[0])

            if iter_id == max(self.next_sync_iter_l):
                ''' testing '''
                global_tensor = torch.tensor([])
                for p in self.global_tensor_list:
                    global_tensor = torch.cat((global_tensor ,p.view(-1)), 0)
                self.connection.root.test(global_tensor, self.round_id)
                
            if iter_id == max(self.next_sync_iter_l):
                self.update_sync_period_l()
                self.logging("update sync_period_l: %s" % (self.sync_period_l))
                self.update_next_sync_iter_l()
            else:
                self.update_next_sync_iter_l(self.layer_list[0])

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_collected_list = [0] * self.receive_size
            self.layer_list = [0] * self.receive_size
            self.first_has_arrived = 0
            self.reporting_index = None

            self.logging('finishing round %d' % self.round_id)
        self.lock.release()


        ''' otherwise, keep spining (blocked) '''
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
            
        # self.logging('return rank %d, round_id %d' % (rank, round_id))
        return self.tensor_list_to_transmit, self.next_sync_iter_l

    def update_d_l(self, layer_list):
        m = self.receive_size
        for index, layer_id in enumerate(layer_list):
            self.d_l[layer_id] = 0
            tau_l = self.sync_period_l[layer_id]
            dim_u_l = self.dim_l[layer_id]
            for i in range(m):
                self.d_l[layer_id] += float((1/m/tau_l/dim_u_l) * ((self.global_tensor_list[layer_id] - self.tensor_collected_list[i][index]).norm()))

    def update_delta_l(self, I):
        accumulated_delta = 0
        for layer_id in I:
            accumulated_delta += self.d_l[layer_id] * self.dim_l[layer_id]
            self.delta_l[layer_id] = accumulated_delta
        for layer_id in I:
            self.delta_l[layer_id] /= accumulated_delta
    
    def update_lambda_l(self, I):
        accumulated_lambda = 0
        for layer_id in I:
            accumulated_lambda += self.dim_l[layer_id]
            self.lambda_l[layer_id] = accumulated_lambda
        for layer_id in I:
            self.lambda_l[layer_id] /= accumulated_lambda

    def update_sync_period_l(self):
        I = numpy.argsort(self.d_l)
        self.update_delta_l(I)
        self.update_lambda_l(I)
        for layer_id in I:
            if self.delta_l[layer_id] < 1-self.lambda_l[layer_id]:
                self.sync_period_l[layer_id] = self.phi * self.init_period
            else:
                self.sync_period_l[layer_id] = self.init_period
    
    def update_next_sync_iter_l(self, layer_list = None):
        if layer_list == None:
            layer_list = range(self.layer_num)
        for layer_id in layer_list:
            self.next_sync_iter_l[layer_id] += self.sync_period_l[layer_id]
    

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
    elif args.interlayer_type == 'GIFT':
        service = GIFT_AggregationService(args.world_size, args.init_period, args.test_process_port)
    elif args.interlayer_type == 'AdaComm':
        service = AdaComm_AggregationService(args.world_size, args.init_period, args.test_process_port)
    elif args.interlayer_type == 'FedLAMA':
        service = FedLAMA_AggregationService(args.world_size, args.init_period, args.test_process_port)
    else:
        service = Default_AggregationService(args.world_size, args.init_period, args.test_process_port)
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    server = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    server.start()