import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time, random, math

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from threading import Lock,Thread

torch.set_printoptions(precision=4)

class FedAvg_AggregationService(Service):
    def __init__(self, world_size, init_period, test_process_port, reporting_fraction):
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
            if round_id % (200//self.init_period) == 0:
                self.connection.root.test(self.averaged_tensor, round_id)

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_collected_list = [0] * self.receive_size
            self.first_has_arrived = 0
            self.reporting_index = None
            
            # # tune
            # if self.round_id == 100:
            #     self.init_period //= 16

            self.logging('finishing round %d' % round_id)
        self.lock.release()


        ''' otherwise, keep spining (blocked) '''
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)
        # self.logging('return rank %d, round_id %d' % (rank, round_id))
        return self.averaged_tensor