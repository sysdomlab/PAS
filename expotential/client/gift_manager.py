import torch
import rpyc
import copy, sys, numpy, math, datetime, pprint, time, os
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist

path=os.path.dirname(__file__)
sys.path.append(path+'/../..')
from model_manager import Model_Manager

CUDA = torch.cuda.is_available()
              

class GIFT_Interlayer():
    def __init__(self, model_manager: Model_Manager, rank, init_period) -> None:
        self.interlayers_type = 'Default'
        self.model_manager = model_manager
        self.model = model_manager.model
        self.round = 0
        self.rank = rank
        self.init_period = init_period
        self.current_period = init_period
        self.last_sync_iter = 0
        self.tensor_of_flattened_params = None
        self.flattened_params_length = 0
        self.sum_num_of_param = 0.0
        self.init_model_structure_related_variables()
        self.logging('create Default Interlayer')

    def init_model_structure_related_variables(self):
        self.tensor_of_flattened_params = self.model_manager.get_flattened_parameter_tensor()
        self.flattened_params_length = len(self.tensor_of_flattened_params)
        self.logging("Param_length: "+str(self.flattened_params_length))

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync_Manager] [GIFT_Interlayer] '+str(string))
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
        self.tensor_of_flattened_params = self.model_manager.get_flattened_parameter_tensor()
        self.sum_num_of_param += self.flattened_params_length/1000000
        if iter_id % max(self.init_period, 64) == 0: self.logging('iter_id: %d; round: %d; sum_num_of_param: %.4f' % (iter_id, self.round, self.sum_num_of_param))
        return self.tensor_of_flattened_params

    def update_period(self):
        self.current_period//=2


class GIFT_Manager():
    """ This object shall be able to run without GIFT (support standard FL by default)"""
    def __init__(self, model_manager: Model_Manager, sync_profile) -> None:
        self.model_manager = model_manager
        self.sync_round_id = 0
        self.rank = sync_profile['rank']
        self.init_period = sync_profile['init_period']
        self.interlayer_type = sync_profile['interlayer_type']
        self.transmit_interlayer = GIFT_Interlayer(self.model_manager, self.rank, self.init_period)

        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect(sync_profile['server_ip'], sync_profile['server_port'], config=rpyc_config)
        self.connection._config['sync_request_timeout']=120

        self.sync_init_model()

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] '+str(string))
        sys.stdout.flush()
    
    def sync_init_model(self):
        self.model_manager.get_param_shape()
        self.tensor_of_flattened_params = self.model_manager.get_flattened_parameter_tensor().cpu()
        tensor_to_restore = self.connection.root.sync_initial_model(self.rank, self.tensor_of_flattened_params)
        tensor_to_restore = rpyc.classic.obtain(tensor_to_restore).cuda() if CUDA else rpyc.classic.obtain(tensor_to_restore)
        self.model_manager.restore_model_with_tensor(tensor_to_restore)

    def try_sync_model(self, iter_id, loss_list):
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
            self.model_manager.restore_model_with_tensor(tensor_received)
            if tune:
                self.transmit_interlayer.update_period()
            return True
        return False