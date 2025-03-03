import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time, random, math

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from threading import Lock,Thread

torch.set_printoptions(precision=4)


class FedLAMA_AggregationService(Service):
    def __init__(self, world_size, init_period, test_process_port, reporting_fraction):
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
                    global_tensor = torch.cat((global_tensor ,p.reshape(-1)), 0)
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