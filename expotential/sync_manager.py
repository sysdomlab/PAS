import torch
import rpyc
import copy, sys, numpy, math, datetime, pprint, time, os
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist


CUDA = torch.cuda.is_available()

InterlayerProfileDict = {
    'Default': None, 
    'GIFT': {
        'ema_theta': 0.9,
        'observe_window_size': 5
    },
    'ParamGIFT': {
        'ema_theta': 0.9,
        'observe_window_size': 5
    },
}


class GIFT_Interlayer():
    def __init__(self, model, sync_frequency, hyperparams, rank):
        self.interlayer_type = 'GIFT'
        self.model = model
        self.sync_frequency = sync_frequency
        self.rank = rank
        self.round = 0
        self.iter_id = 0
        self.sum_number_of_param = 0.0
        

        """ Set hyper_parameters """
        self.ema_theta = hyperparams['ema_theta']
        self.observe_window_size = hyperparams['observe_window_size'] * sync_frequency 
        """ Initialize model-structure-related variables """
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.tensor_of_flattened_params = None
        self.last_tensor_of_flattened_params = None
        self.flattened_params_length = -1 #
        self.init_model_structure_related_variables()

        """ Initialize GIFT-related variables """
        self.next_transmit_id = sync_frequency
        if self.rank == 0:
            self.grad_consistency = 1
            self.window_start_iter = 0
            self.positive_grad_ema = torch.zeros(self.flattened_params_length).cuda() if CUDA else torch.zeros(self.flattened_params_length)
            self.negative_grad_ema = torch.zeros(self.flattened_params_length).cuda() if CUDA else torch.zeros(self.flattened_params_length)
        self.logging('create GIFT Interlayer')

    def init_model_structure_related_variables(self):
        s_index = 0
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for p in self.model.parameters():
            self.param_shape_list.append(p.data.shape)
            flattened_param = p.data.view(-1)
            e_index = s_index + len(flattened_param)
            self.cutoff_index_list.append([s_index, e_index])
            s_index = e_index
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param),0)
        self.flattened_params_length = len(self.tensor_of_flattened_params)
        self.logging("Param_length: "+str(self.flattened_params_length))
        self.last_tensor_of_flattened_params = copy.deepcopy(self.tensor_of_flattened_params) if self.rank == 0 else None

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync_Manager] [GIFT_Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
        sys.stdout.flush()

    def generate_flattened_tensor_to_transmit(self, iter_id):
        """"""
        self.iter_id = iter_id
        if iter_id != self.next_transmit_id:
            return torch.tensor([])
        # self.model = self.model.cpu() if CUDA else self.model # "tcp" can support gpu
        self.round += 1
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for p in self.model.parameters():
            flattened_param = p.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param),0)
        self.sum_number_of_param += self.flattened_params_length / 1000000
        return self.tensor_of_flattened_params

    def restore_model_from_tensor_received(self, tensor_received):
        """"""
        for i, param in enumerate(self.model.parameters()):
            param.data = tensor_received[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])
        if self.rank == 0 and self.iter_id % 128 == 0:
            self.last_tensor_of_flattened_params = copy.deepcopy(tensor_received)
        frequency_tensor = torch.tensor(self.sync_frequency)
        dist.broadcast(frequency_tensor, src=0)
        self.sync_frequency = int(frequency_tensor)
        self.next_transmit_id += self.sync_frequency
        # self.model = self.model.cuda() if CUDA else self.model

    def calculate_grad_consistency(self, tensor_list_received):
        """"""
        positive_grad = torch.zeros(self.flattened_params_length).cuda() if CUDA else torch.zeros(self.flattened_params_length)
        negative_grad = torch.zeros(self.flattened_params_length).cuda() if CUDA else torch.zeros(self.flattened_params_length)
        zeros = torch.zeros(self.flattened_params_length).cuda() if CUDA else torch.zeros(self.flattened_params_length)
        for tensor_received in tensor_list_received:
            if CUDA:
                tensor_received = tensor_received.cuda()
            grad_tensor = tensor_received - self.last_tensor_of_flattened_params
            positive_grad += torch.where(grad_tensor >= 0, grad_tensor, zeros)
            negative_grad -= torch.where(grad_tensor < 0, grad_tensor, zeros)
        
        '''smoothing'''
        self.positive_grad_ema = self.ema_theta*self.positive_grad_ema + (1-self.ema_theta)*positive_grad
        self.negative_grad_ema = self.ema_theta*self.negative_grad_ema + (1-self.ema_theta)*negative_grad
        grad_consistency = (abs(self.positive_grad_ema-self.negative_grad_ema) / (self.positive_grad_ema+self.negative_grad_ema)).sum() / self.flattened_params_length
        self.logging("iter_id: %d; round: %d; averaging grad_consistency: %.4f; sum_num_of_param: %.4f" % (self.iter_id, self.round, grad_consistency, self.sum_number_of_param))
        self.update_frequency(grad_consistency) 


    def update_frequency(self, grad_consistency):
        if (grad_consistency < self.grad_consistency) or (grad_consistency - self.grad_consistency > 0.03):
            self.grad_consistency = grad_consistency
            self.window_start_iter = self.iter_id
        else:
            if self.iter_id > self.window_start_iter + self.observe_window_size and self.sync_frequency > 1: 
                self.sync_frequency //= 2
                self.window_start_iter = self.iter_id
                self.grad_consistency = 1
                self.positive_grad_ema = torch.zeros(self.flattened_params_length).cuda() if CUDA else torch.zeros(self.flattened_params_length)
                self.negative_grad_ema = torch.zeros(self.flattened_params_length).cuda() if CUDA else torch.zeros(self.flattened_params_length)
                self.logging("sync_frequency: %d" % self.sync_frequency)
              

class Default_Interlayer():
    def __init__(self, model, rank, init_period) -> None:
        self.interlayers_type = 'Default'
        self.model = model
        self.round = 0
        self.rank = rank
        self.init_period = init_period
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.tensor_of_flattened_params = None
        self.flattened_params_length = 0
        self.sum_num_of_param = 0.0
        self.init_model_structure_related_variables()
        self.logging('create Default Interlayer')

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
        self.flattened_params_length = len(self.tensor_of_flattened_params)
        self.logging("Param_length: "+str(self.flattened_params_length))

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync_Manager] [Default_Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
        sys.stdout.flush()

    def generate_flattened_tensor_to_transmit(self, iter_id):
        """"""
        if iter_id % self.init_period != 0:
            return torch.tensor([])   #低版本pytorch不接受none
        # self.model = self.model.cpu() if CUDA else self.model # "tcp" can support gpu
        self.round += 1
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for p in self.model.parameters():
            flattened_param = p.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param),0)
        self.sum_num_of_param += self.flattened_params_length/1000000
        if iter_id % max(self.init_period, 64) == 0: self.logging('iter_id: %d; round: %d; sum_num_of_param: %.4f' % (iter_id, self.round, self.sum_num_of_param))
        return self.tensor_of_flattened_params

    def restore_model_from_tensor_received(self, tensor_received):
        """"""
        for i, param in enumerate(self.model.parameters()):
            param.data = tensor_received[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])


class Sync_Manager():
    """ This object shall be able to run without GIFT (support standard FL by default)"""
    def __init__(self, model, sync_profile) -> None:
        self.model = model
        self.sync_round_id = 0
        self.rank = sync_profile['rank']
        self.init_period = sync_profile['init_period']
        self.interlayer_type = sync_profile['interlayer_type']
        self.transmit_interlayer = self.create_transmit_interlayer(sync_profile['interlayer_type'])

        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect(sync_profile['server_ip'], sync_profile['server_port'], config=rpyc_config)

        self.sync_init_model()

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] '+str(string))
        sys.stdout.flush()

    def create_transmit_interlayer(self, interlayer_type):
        # self.model = self.model.cpu() if CUDA else self.model # "tcp" can support gpu
        if interlayer_type == 'Default':
            transmit_interlayer = Default_Interlayer(self.model, self.rank, self.init_period)
        return transmit_interlayer

    def sync_init_model(self):
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for p in self.model.parameters():
            flattened_param = p.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param),0)
        if CUDA: self.tensor_of_flattened_params = self.tensor_of_flattened_params.cpu()
        tensor_to_restore = self.connection.root.sync_initial_model(self.rank, self.tensor_of_flattened_params)
        tensor_to_restore = rpyc.classic.obtain(tensor_to_restore).cuda() if CUDA else rpyc.classic.obtain(tensor_to_restore)
        self.transmit_interlayer.restore_model_from_tensor_received(tensor_to_restore)

    def try_sync_model(self, iter_id):
        """ Conduct remote synchronization. """
        t1 = time.time()
        flattened_tensor_to_transmit = self.transmit_interlayer.generate_flattened_tensor_to_transmit(iter_id)
        if len(flattened_tensor_to_transmit) != 0:
            t2 = time.time()
            self.logging('generate_time: %.4f' % (t2-t1))
            self.sync_round_id += 1
            if CUDA: flattened_tensor_to_transmit = flattened_tensor_to_transmit.cpu()
            t1 = time.time()
            tensor_received = self.connection.root.aggregate(self.rank, self.sync_round_id, flattened_tensor_to_transmit)
            t2 = time.time()
            tensor_received = rpyc.classic.obtain(tensor_received).cuda() if CUDA else rpyc.classic.obtain(tensor_received)
            t3 = time.time()
            self.logging('sync_time: %.4f, obtain_time: %.4f' % (t3-t1,t3-t2))
            self.transmit_interlayer.restore_model_from_tensor_received(tensor_received)
            return True
        return False