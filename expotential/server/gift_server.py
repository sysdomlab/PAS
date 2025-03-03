import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time, random, math

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from threading import Lock,Thread

torch.set_printoptions(precision=4)

GIFT_hyperparams_profile = {
    'ema_theta' : 0.8,
    'observe_window_size' : 3,
    'tune_threshold' : 0.2,
}


class GIFT_AggregationService(Service):
    def __init__(self, world_size, init_period, test_process_port, reporting_fraction):
        self.world_size = world_size
        self.init_period = init_period
        self.current_period = init_period
        self.receive_size = int(world_size*reporting_fraction)
        self.round_id = 0
        self.flattened_params_length = 0
        self.lock = Lock()
        self.ema_theta = GIFT_hyperparams_profile['ema_theta']
        self.observe_window_size = GIFT_hyperparams_profile['observe_window_size']
        self.tune_threshold = GIFT_hyperparams_profile['tune_threshold']
        self.min_consistency = 1
        self.avg_consistency_window_start_round = 0
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
                # self.tune = self.update_period_by_avg()
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
        zero_mask = ((self.ema_positive_grad+self.ema_negative_grad) == 0)
        grad_consistency[zero_mask] = 0

        mask2 = ((grad_consistency > self.min_grad_consistency) \
                 & (grad_consistency<self.tune_threshold) & (self.round_id > self.window_start_round + self.observe_window_size))
        self.tune_tag[mask2] = 1


        mask1 = ((grad_consistency <= self.min_grad_consistency) | (grad_consistency > self.tune_threshold))
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
    
    def update_period_by_avg(self):
        grad_consistency = abs(self.ema_positive_grad-self.ema_negative_grad) / (self.ema_positive_grad+self.ema_negative_grad)
        zero_mask = ((self.ema_positive_grad+self.ema_negative_grad) == 0)
        grad_consistency[zero_mask] = 0

        average_consistency = grad_consistency.sum()/self.flattened_params_length
        self.logging("averaging gradient consistency: %.4f" % (average_consistency))
        if (average_consistency < self.min_consistency):
            self.min_consistency = average_consistency
            self.avg_consistency_window_start_round = self.round_id
        if self.round_id > self.avg_consistency_window_start_round + self.observe_window_size and self.current_period > self.init_period//16:
            self.current_period //= 2
            self.avg_consistency_window_start_round = self.round_id
            self.observe_window_size *= 2
            self.min_consistency = 1
            self.ema_positive_grad = torch.zeros(self.flattened_params_length)
            self.ema_negative_grad = torch.zeros(self.flattened_params_length)
            self.logging('sync period tune to %d' % self.current_period)
            return 1
        return 0