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

        self.tensor_of_flattened_params = None
        self.freezing_tensor_of_flattened_params = None
        self.tensor_list_to_transmit = []
        self.tensor_list_received = None
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.layer_len_list = []
        self.layer_num=0

        ''' connect to the FL server '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect('localhost', sync_process_port, config=rpyc_config)
        self.connection._config['sync_request_timeout']=120
        # self.sync_fun = rpyc.async_(self.connection.root.sync)

        ''' init model '''
        self.init_model_structure_related_variables()

        self.layer_sync_period = [init_period] * self.layer_num

        self.period_list = []
        temp_period = init_period
        while temp_period >= 1:
            self.period_list.append(temp_period)
            temp_period//=2
        self.logging('-- Expotential CAS ')


    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync_Manager] [CAS_Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
        sys.stdout.flush()

    def init_model_structure_related_variables(self):
        s_index = 0
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        self.tensor_list_to_transmit = []
        for param in self.model.parameters():
            self.param_shape_list.append(param.data.shape)
            flatten_params = param.data.view(-1)
            self.layer_len_list.append(len(flatten_params))
            e_index = s_index + len(flatten_params)
            self.cutoff_index_list.append([s_index, e_index])
            s_index = e_index
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flatten_params),0)
            self.tensor_list_to_transmit.append(param.data.cpu())
            self.layer_num += 1
        tensor_list_received = self.connection.root.sync_initial_model(self.tensor_list_to_transmit, self.init_period) \
            if CUDA else self.connection.root.sync_initial_model(self.tensor_list_to_transmit, self.init_period)
        self.tensor_list_received = [rpyc.classic.obtain(i).cuda() for i in tensor_list_received] if CUDA \
            else [rpyc.classic.obtain(i) for i in tensor_list_received]
        self.restore_model_from_tensor_received()
        self.flattened_params_length = len(self.tensor_of_flattened_params)
        self.logging("Param_length: "+str(self.flattened_params_length))

    def generate_flattened_tensor_to_transmit(self, iter_id):
        """"""
        round_iter = iter_id - self.last_round_iter
        if round_iter == self.global_period: 
            self.last_round_iter = iter_id
        self.iter_id = iter_id
        if round_iter == 1: 
            self.round_id += 1
        if round_iter not in self.period_list:
            return 0
        if round_iter not in self.layer_sync_period:
            return 0
        
        sync_layer_id = []
        for i in range(self.layer_num):
            if self.layer_sync_period[i] == round_iter:
                sync_layer_id.append(i)
                self.sync_params_volume += self.layer_len_list[i]/1000000
        layer_id = 0
        self.tensor_list_to_transmit = []
        for param in self.model.parameters():
            if self.layer_sync_period[layer_id] == round_iter:
                self.tensor_list_to_transmit.append(param.data.cpu())
            layer_id += 1
    
        t1 = time.time()
        # if round_iter == self.global_period and (self.round_id - self.last_update_round) == (self.tune_times):
        if round_iter == self.global_period:
            self.logging('iter_id: %d; round: %d; sum_num_of_param: %.4f' % (self.iter_id, self.round_id, self.sync_params_volume))
            self.connection.root.sync_and_update(self.tensor_list_to_transmit, sync_layer_id, round_iter)
        else:
            self.connection.root.sync(self.tensor_list_to_transmit, sync_layer_id, round_iter)
        # self.sync_fun(transmit_list, round_iter)
        t2 = time.time()
        self.logging('put_time: %.4f' % (t2-t1))
        if round_iter == self.global_period:
            tensor_list_received = self.connection.root.get_restore_tensor()
            self.tensor_list_received = [rpyc.classic.obtain(i).cuda() for i in tensor_list_received] if CUDA \
            else [rpyc.classic.obtain(i) for i in tensor_list_received]
            return 1
        else:
            return 0
              
    def update_period(self):
        index = self.connection.root.update_period()
        index = rpyc.classic.obtain(index)
        for i in index:
            self.layer_sync_period[i] //= 2
        if self.global_period not in self.layer_sync_period:
            self.tune_global_period()

    def tune_global_period(self):
        self.logging('tune global period')
        self.global_period //= 2
        self.tune_times *= 2

    def restore_model_from_tensor_received(self):
        """"""
        for i, param in enumerate(self.model.parameters()):
            # Note that you are actually assigning a variable to itself
            param.data = self.tensor_list_received[i]


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
        

    def try_sync_model(self, iter_id):
        """ Conduct remote synchronization. """
        t1 = time.time()
        if self.transmit_interlayer.generate_flattened_tensor_to_transmit(iter_id) == 0:
            return False
        t2 = time.time()
        self.logging('generate_time: %.4f' % (t2-t1))
        self.sync_round_id += 1
        # if (self.sync_round_id-self.transmit_interlayer.last_update_round) == (self.transmit_interlayer.tune_times):
        self.transmit_interlayer.update_period()
        self.transmit_interlayer.restore_model_from_tensor_received()
        return True