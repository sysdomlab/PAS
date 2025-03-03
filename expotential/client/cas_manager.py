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


APF_hyperparams = {
    'freezing_check_frequency' : 1, # unit for freezing_frequency is 'round'
    'ema_alpha' : 0.95,
    'stable_threshold' : 0.05,
    'enable_random_freezing' : False
}


class CAS_Interlayer():
    def __init__(self, model_manager: Model_Manager, init_period, rank, sync_process_port) -> None:
        ''' sync_variable '''
        self.model_manager = model_manager
        self.model = model_manager.model
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
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.t_round_start = 0

        ''' connect to the FL server '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect('localhost', sync_process_port, config=rpyc_config)
        self.connection._config['sync_request_timeout']=1200
        # self.sync_fun = rpyc.async_(self.connection.root.sync)

        ''' init model '''
        self.init_model_structure_related_variables()

        self.param_sync_period = torch.full([self.flattened_params_length], self.global_period, dtype=torch.int32).cuda() if CUDA \
                                else torch.full([self.flattened_params_length], self.global_period, dtype=torch.int32)
        self.mask = torch.zeros(self.flattened_params_length, dtype=torch.bool).cuda() if CUDA \
                    else torch.zeros(self.flattened_params_length, dtype=torch.bool)


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
        self.model_manager.get_param_shape()
        self.tensor_of_flattened_params = self.model_manager.get_flattened_parameter_tensor()
        tensor_received = self.connection.root.sync_initial_model(self.tensor_of_flattened_params.cpu(), self.init_period)
        tensor_received = rpyc.classic.obtain(tensor_received).cuda() if CUDA else rpyc.classic.obtain(tensor_received)
        self.model_manager.restore_model_with_tensor(tensor_received)
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
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        self.mask = (self.param_sync_period == round_iter).cuda() if CUDA else (self.param_sync_period == round_iter)
        length = len(self.mask.nonzero())
        if length == 0: return 0

        self.sync_params_volume += length/1000000
        self.tensor_of_flattened_params = self.model_manager.get_flattened_parameter_tensor()
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
            tensor_received = self.connection.root.get_restore_tensor(self.flattened_params_length)
            tensor_received = rpyc.classic.obtain(tensor_received).cuda() if CUDA else rpyc.classic.obtain(tensor_received)
            self.model_manager.restore_model_with_tensor(tensor_received)
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


    def tune_global_period(self):
        mask = (self.param_sync_period == self.global_period)
        self.logging('tune global period')
        self.global_period = int(self.global_period/self.divisor)
        self.param_sync_period[mask] = (self.param_sync_period[mask]/self.divisor).type(torch.int32)



class CASAPF_Interlayer(CAS_Interlayer):
    def __init__(self, model_manager: Model_Manager, init_period, rank, sync_process_port) -> None:
        ''' sync_variable '''
        self.model_manager = model_manager
        self.model = model_manager.model
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
        self.received_tensor = None
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.t_round_start = time.time()

        ''' APF hyperparameters '''
        self.apf_ema_alpha = APF_hyperparams['ema_alpha']
        self.stable_threshold = APF_hyperparams['stable_threshold'] # this shall be larger than 1-self.ema_alpha
        self.tighter_stable_criterion_threshold = 0.8
        self.enable_random_freezing = APF_hyperparams['enable_random_freezing']

        ''' connect to the FL server '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect('localhost', sync_process_port, config=rpyc_config)
        self.connection._config['sync_request_timeout']=1200
        # self.sync_fun = rpyc.async_(self.connection.root.sync)

        ''' init model '''
        self.init_model_structure_related_variables()

        ''' cas variable '''
        self.param_sync_period = torch.full([self.flattened_params_length], self.global_period, dtype=torch.int32).cuda() if CUDA \
                                else torch.full([self.flattened_params_length], self.global_period, dtype=torch.int32)

        self.period_list = []
        temp_period = init_period
        while temp_period >= 1:
            self.period_list.append(temp_period)
            temp_period = int(temp_period/self.divisor)

        ''' apf variable '''
        self.active_mask = torch.ones(self.flattened_params_length).bool().cuda() if CUDA \
            else torch.ones(self.flattened_params_length).bool()
        self.grad_ema = torch.zeros(self.flattened_params_length).cuda() if CUDA \
            else torch.zeros(self.flattened_params_length)
        self.abs_grad_ema = torch.zeros(self.flattened_params_length).cuda() if CUDA \
            else torch.zeros(self.flattened_params_length)
        self.freezing_lengths = torch.zeros(self.flattened_params_length).int().cuda() if CUDA \
            else torch.zeros(self.flattened_params_length).int()
        self.fc_round_ids_to_unfreeze_params = torch.zeros(self.flattened_params_length).int().cuda() if CUDA \
            else torch.zeros(self.flattened_params_length).int()
        self.last_tensor_of_flattened_params = copy.deepcopy(self.tensor_of_flattened_params)
        
        self.logging('CAS-APF ')


    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync_Manager] [CASAPF_Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
        sys.stdout.flush()

    def init_model_structure_related_variables(self):
        self.model_manager.get_param_shape()
        self.tensor_of_flattened_params = self.model_manager.get_flattened_parameter_tensor()
        tensor_to_restore = self.connection.root.sync_initial_model(self.tensor_of_flattened_params.cpu(), self.init_period)
        self.received_tensor = rpyc.classic.obtain(tensor_to_restore).cuda() if CUDA else rpyc.classic.obtain(tensor_to_restore)
        self.model_manager.restore_model_with_tensor(self.received_tensor)
        self.flattened_params_length = len(self.tensor_of_flattened_params)
        self.tensor_of_flattened_params = self.received_tensor
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
        
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA \
            else torch.tensor([])
        mask = ((self.param_sync_period == round_iter) & self.active_mask)
        length = int(mask.sum())
        if length == 0: return 0

        self.sync_params_volume += length/1000000
        self.tensor_of_flattened_params = self.model_manager.get_flattened_parameter_tensor()

        t1 = time.time()
        # if round_iter == self.global_period and (self.round_id - self.last_update_round) == (self.tune_times):
        if round_iter == self.global_period:
            self.logging('iter_id: %d; round: %d; sum_num_of_param: %.4f' % (self.iter_id, self.round_id, self.sync_params_volume))
            self.connection.root.sync_and_update([self.tensor_of_flattened_params[mask].cpu()], mask.cpu(), round_iter)
        else:
            self.connection.root.sync([self.tensor_of_flattened_params[mask].cpu()], mask.cpu(), round_iter)
        # self.sync_fun(transmit_list, round_iter)
        t2 = time.time()
        self.logging('put_time: %.4f' % (t2-t1))
        if round_iter == self.global_period:
            received_tensor = self.connection.root.get_restore_tensor(int(self.active_mask.sum()))
            self.received_tensor = rpyc.classic.obtain(received_tensor).cuda() if CUDA else rpyc.classic.obtain(received_tensor)
            self.model_manager.restore_model_with_tensor(self.received_tensor)
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


    def tune_global_period(self):
        mask = (self.param_sync_period == self.global_period)
        self.logging('tune global period')
        self.global_period = int(self.global_period/self.divisor)
        self.param_sync_period[mask] = (self.param_sync_period[mask]/self.divisor).type(torch.int32)
    

    def update_param_freezing_periods_in_TCP_manner(self):
        """ update the active mask of each parameter based on global gradient stability. """
        ''' Calculate the effective stepping rate of each parameter (in tensor mode for fast processing). '''
        # torch.save(self.received_tensor, f"/home/zgan/CASP-code/Logs/rank{self.rank}_round{self.round_id}_recv.pt")
        # torch.save(self.last_tensor_of_flattened_params, f"/home/zgan/CASP-code/Logs/rank{self.rank}_round{self.round_id}_recv.pt")
        
        grad = self.received_tensor[self.active_mask] - self.last_tensor_of_flattened_params[self.active_mask]
        self.grad_ema[self.active_mask] = self.grad_ema[self.active_mask] * self.apf_ema_alpha + grad * (1.0 - self.apf_ema_alpha) 
        self.abs_grad_ema[self.active_mask] = self.abs_grad_ema[self.active_mask] * self.apf_ema_alpha + torch.abs(grad) * (1.0 - self.apf_ema_alpha)
        effective_stepping_rate = torch.abs(self.grad_ema[self.active_mask]) / self.abs_grad_ema[self.active_mask]
        self.logging("effective_stepping_rate: {:.4f}".format(float(torch.mean(effective_stepping_rate))))

        ''' Update the freezing period length and also the should-be-unfrozen frequency_check_round_id based on effective stepping rate of each param. '''
        self.freezing_lengths[self.active_mask] = torch.where(effective_stepping_rate < self.stable_threshold, 
                                                               self.freezing_lengths[self.active_mask]+1, 
                                                               self.freezing_lengths[self.active_mask]//2)
        self.fc_round_ids_to_unfreeze_params[self.active_mask] = self.round_id + self.freezing_lengths[self.active_mask]
        self.last_tensor_of_flattened_params = copy.deepcopy(self.received_tensor)

    def refresh_freezing_status(self):
        """ Called after each global synchronization is conducted. Update the freezing status and related statistics."""
        ''' Update the freezing period in a TCP manner and then update the freezing mask accordingly '''
        self.update_param_freezing_periods_in_TCP_manner()
        self.active_mask = (self.fc_round_ids_to_unfreeze_params == self.round_id)
        ''' Record the stable-ratio statistics and adaptively tune stability threshold if necessary '''
        self.freezing_ratio = 1 - self.active_mask.sum().float() / self.flattened_params_length
        self.logging('current stable ratio: %.4f' % self.freezing_ratio)
        # if self.freezing_ratio > self.tighter_stable_criterion_threshold:
        #     self.stable_threshold /= 2.0
        #     self.logging('make stable criterion tighter')



class CAS_Manager():
    def __init__(self, model_manager: Model_Manager, sync_profile) -> None:
        self.model_manager = model_manager
        self.sync_round_id = 0
        self.init_period = sync_profile['init_period']
        self.interlayer_type = sync_profile['interlayer_type']
        self.rank = sync_profile['rank']
        if sync_profile['interlayer_type'] == 'CAS':
            self.transmit_interlayer = CAS_Interlayer(self.model_manager, self.init_period, self.rank, sync_profile['sync_process_port'])
        else:
            self.transmit_interlayer = CASAPF_Interlayer(self.model_manager, self.init_period, self.rank, sync_profile['sync_process_port'])


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
        if self.interlayer_type == 'CAS-APF':
            self.transmit_interlayer.refresh_freezing_status()
        t2 = time.time()
        self.logging('update_period_time: %.4f' % (t2-t1))
        return True