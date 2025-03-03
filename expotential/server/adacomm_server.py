import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time, random, math

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from threading import Lock,Thread

from fedavg_server import FedAvg_AggregationService


torch.set_printoptions(precision=4)




class AdaComm_AggregationService(FedAvg_AggregationService):
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