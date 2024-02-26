import torch
import rpyc
import copy, sys, numpy, math, datetime, pprint, time, io
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist

CUDA = torch.cuda.is_available()


class CAS_Interlayer():
    def __init__(self, model, init_period, rank, sync_process_port) -> None:
        ''' sync_variable '''
        self.model = model
        self.rank = rank
        self.iter_id = 0
        self.round_id = 0
        self.flattened_params_length = 0
        self.global_period = init_period
        self.init_period = init_period
        # self.tune_cell = int(init_period//5)
        self.last_round_iter = 0
        self.sync_params_volume = 0
        self.last_update_round = 0
        self.tune_times = int(1)
        self.divisor = 2

        self.tensor_of_flattened_params = None
        self.freezing_tensor_of_flattened_params = None
        self.received_tensor = None
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.t_round_start = 0

        ''' connect to the FL server '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect('localhost', sync_process_port, config=rpyc_config)
        self.connection._config['sync_request_timeout']=120
        # self.sync_fun = rpyc.async_(self.connection.root.sync)

        ''' init model '''
        self.init_model_structure_related_variables()

        self.param_sync_period = torch.full([self.flattened_params_length], self.global_period, dtype=torch.int32).cuda() if CUDA \
                                else torch.full([self.flattened_params_length], self.global_period, dtype=torch.int32)
        self.mask = torch.zeros(self.flattened_params_length, dtype=torch.bool).cuda() if CUDA \
                    else torch.zeros(self.flattened_params_length, dtype=torch.bool)
        self.freezing_mask = copy.deepcopy(self.mask)


        self.period_list = []
        temp_period = init_period
        while temp_period >= 1:
            self.period_list.append(temp_period)
            temp_period = int(temp_period/self.divisor)
        self.logging('-- Expotential CAS ')


    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync_Manager] [CAS_Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
        sys.stdout.flush()

    def init_model_structure_related_variables(self):
        s_index = 0
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for param in self.model.parameters():
            self.param_shape_list.append(param.data.shape)
            flatten_params = param.data.view(-1)
            e_index = s_index + len(flatten_params)
            self.cutoff_index_list.append([s_index, e_index])
            s_index = e_index
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flatten_params),0)
        tensor_to_restore = self.connection.root.sync_initial_model(self.tensor_of_flattened_params.cpu(), self.init_period) \
            if CUDA else self.connection.root.sync_initial_model(self.tensor_of_flattened_params, self.init_period)
        self.received_tensor = rpyc.classic.obtain(tensor_to_restore).cuda() if CUDA else rpyc.classic.obtain(tensor_to_restore)
        self.restore_model_from_tensor_received()
        self.flattened_params_length = len(self.tensor_of_flattened_params)
        self.logging("Param_length: "+str(self.flattened_params_length))
        self.freezing_tensor_of_flattened_params = copy.deepcopy(self.tensor_of_flattened_params)

    def generate_flattened_tensor_to_transmit(self, iter_id):
        """"""
        round_iter = iter_id - self.last_round_iter
        if round_iter == self.global_period: 
            self.last_round_iter = iter_id
        self.iter_id = iter_id
        if round_iter == 1: 
            self.freezing_mask.fill_(0)
            self.round_id += 1
        if round_iter not in self.period_list:
            return 0
        # if len(self.freezing_mask.nonzero()) != 0:
        #     self.freezing()
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        self.mask = (self.param_sync_period == round_iter).cuda() if CUDA else (self.param_sync_period == round_iter)
        length = len(self.mask.nonzero())
        if length == 0: return 0

        self.sync_params_volume += length/1000000
        for param in self.model.parameters():
            flattened_param = param.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param), 0)
        self.freezing_tensor_of_flattened_params[self.mask] = self.tensor_of_flattened_params[self.mask]
        self.freezing_mask += self.mask
        t1 = time.time()
        # if round_iter == self.global_period and (self.round_id - self.last_update_round) == (self.tune_times):
        if round_iter == self.global_period:
            self.logging('iter_id: %d; round: %d; sum_num_of_param: %.4f' % (self.iter_id, self.round_id, self.sync_params_volume))
            self.connection.root.sync_and_update([self.tensor_of_flattened_params[self.mask].cpu()], self.mask.cpu(), round_iter) \
            if CUDA else self.connection.root.sync_and_update([self.tensor_of_flattened_params[self.mask]], self.mask, round_iter)
        else:
            self.connection.root.sync([self.tensor_of_flattened_params[self.mask].cpu()], self.mask.cpu(), round_iter) \
            if CUDA else self.connection.root.sync([self.tensor_of_flattened_params[self.mask]], self.mask, round_iter)
        # self.sync_fun(transmit_list, round_iter)
        t2 = time.time()
        self.logging('put_time: %.4f' % (t2-t1))
        if round_iter == self.global_period:
            t_round_end = time.time()
            # self.logging('round_computation_time=%.4f' % (t_round_end-self.t_round_start))
            received_tensor = self.connection.root.get_restore_tensor()
            self.received_tensor = rpyc.classic.obtain(received_tensor).cuda() if CUDA else rpyc.classic.obtain(received_tensor)
            self.t_round_start = time.time()
            return 1
        else:
            return 0
        
        
    def update_period(self):
        index = self.connection.root.update_period()
        index = rpyc.classic.obtain(index)
        global_tune_tag = index[-1:]
        index = index[:-1].cuda() if CUDA else index[:-1]
        self.param_sync_period[index] = (self.param_sync_period[index]/self.divisor).type(torch.int32)
        if global_tune_tag == 1:
            self.tune_global_period()


    def freezing(self):
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for param in self.model.parameters():
            flattened_param = param.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param), 0)
        self.tensor_of_flattened_params[self.freezing_mask] = self.freezing_tensor_of_flattened_params[self.freezing_mask]
        for i, param in enumerate(self.model.parameters()):
            param.data = self.tensor_of_flattened_params[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])


    def tune_global_period(self):
        mask = (self.param_sync_period == self.global_period)
        self.logging('tune global period')
        self.global_period = int(self.global_period/self.divisor)
        self.param_sync_period[mask] = (self.param_sync_period[mask]/self.divisor).type(torch.int32)

    def restore_model_from_tensor_received(self):
        """"""
        for i, param in enumerate(self.model.parameters()):
            param.data = self.received_tensor[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])


class CAS_Manager():
    def __init__(self, model, sync_profile) -> None:
        self.model = model
        self.sync_round_id = 0
        self.init_period = sync_profile['init_period']
        self.interlayer_type = sync_profile['interlayer_type']
        self.rank = sync_profile['rank']
        self.transmit_interlayer = CAS_Interlayer(self.model, self.init_period, self.rank, sync_profile['sync_process_port'])


    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [CAS Manager] '+str(string))
        sys.stdout.flush()
        

    def try_sync_model(self, iter_id, loss_list):
        """ Conduct remote synchronization. """
        t1 = time.time()
        if self.transmit_interlayer.generate_flattened_tensor_to_transmit(iter_id) == 0:
            return False
        t2 = time.time()
        self.logging('generate_time: %.4f' % (t2-t1))
        self.sync_round_id += 1
        # if (self.sync_round_id-self.transmit_interlayer.last_update_round) == (self.transmit_interlayer.tune_times):
        t1 = time.time()
        self.transmit_interlayer.update_period()
        t2 = time.time()
        self.logging('update_period_time: %.4f' % (t2-t1))
        self.transmit_interlayer.last_update_round = self.sync_round_id
        self.transmit_interlayer.restore_model_from_tensor_received()
        return True