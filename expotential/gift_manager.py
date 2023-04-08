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
    def __init__(self, model, rank, init_period) -> None:
        self.interlayers_type = 'Default'
        self.model = model
        self.round = 0
        self.rank = rank
        self.init_period = init_period
        self.current_period = init_period
        self.last_sync_iter = 0
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
        if (iter_id-self.last_sync_iter) != self.current_period:
            return torch.tensor([])   #低版本pytorch不接受none
        # self.model = self.model.cpu() if CUDA else self.model # "tcp" can support gpu
        self.round += 1
        self.last_sync_iter = iter_id
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

    def update_period(self):
        self.current_period//=2


class GIFT_Manager():
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
        transmit_interlayer = GIFT_Interlayer(self.model, self.rank, self.init_period)
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
            list_received = self.connection.root.aggregate(self.rank, self.sync_round_id, flattened_tensor_to_transmit)
            t2 = time.time()
            list_received = rpyc.classic.obtain(list_received)
            tune = list_received[1]
            tensor_received = list_received[0].cuda() if CUDA else list_received[0]
            t3 = time.time()
            self.logging('sync_time: %.4f, obtain_time: %.4f' % (t3-t1,t3-t2))
            self.transmit_interlayer.restore_model_from_tensor_received(tensor_received)
            if tune:
                self.transmit_interlayer.update_period()
            return True
        return False