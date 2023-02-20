import torch
import rpyc
import copy, sys, numpy, math, datetime, pprint, time, io
import pickle
import torch.multiprocessing as mp
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist

mp.set_start_method('spawn', force=True)
print(mp.get_start_method())
CUDA = torch.cuda.is_available()

ParamGIFTProfile = {
    'ema_theta': 0.9,
    'observe_window_size': 5,
}


class Sync_Process(mp.Process):
    def __init__(self, server_ip, server_port, rank, init_period, transmit_queue, period_queue, restore_queue, mask_queue, update_period_tag, update_index_queue) -> None:
        super(Sync_Process, self).__init__()
        ''' sync variables '''
        self.rank = rank
        self.round_id = 1
        self.round_transmit_param_volume = 0
        self.flattened_params_length = 0
        self.flattened_tenosr_to_restore = None
        self.mask = None
        self.server_ip = server_ip
        self.server_port = server_port
        self.init_period = init_period

        ''' queues(processes common memory) '''
        self.transmit_queue = transmit_queue
        self.period_queue = period_queue
        self.restore_queue = restore_queue
        self.mask_queue = mask_queue
        self.update_period_tag = update_period_tag
        self.update_index_queue = update_index_queue

        self.logging(' create sync process successfully ')

    def run(self):
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect(self.server_ip, self.server_port, config=rpyc_config)

        self.sync_initial_model(self.init_period)
        while True:
            if not self.transmit_queue.empty():
                self.sync()
            if not self.update_period_tag.empty():
                self.update_period_tag.get()
                self.update_period()
            time.sleep(0.01)

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync_Manager] [Sync_Process] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
        sys.stdout.flush()

    def sync_initial_model(self, init_period):
        tensor_to_transmit = self.transmit_queue.get()
        self.flattened_params_length = len(tensor_to_transmit)
        tensor_to_restore = self.connection.root.sync_initial_model(self.rank, init_period, tensor_to_transmit)
        tensor_to_restore = rpyc.classic.obtain(tensor_to_restore)
        self.flattened_tenosr_to_restore = copy.deepcopy(tensor_to_restore)
        self.mask = torch.zeros(self.flattened_params_length, dtype=torch.bool)
        self.restore_queue.put(tensor_to_restore)

    def sync(self):
        ''' aggregate '''
        t1 = time.time()
        tensor_to_transmit = self.transmit_queue.get()
        t2 = time.time()
        self.logging('get_time: %.4f' % (t2-t1))
        temp_period = self.period_queue.get()
        temp_len = len(tensor_to_transmit)
        t1 = time.time()
        tensor_to_restore = self.connection.root.aggregate(self.rank, self.round_id, tensor_to_transmit, temp_period)
        tensor_to_restore = rpyc.classic.obtain(tensor_to_restore)
        t2 = time.time()
        self.logging('inner_sync_time: %.4f' % (t2-t1))
        ''' restore '''
        self.mask = self.mask_queue.get()
        self.flattened_tenosr_to_restore[self.mask] = tensor_to_restore
        self.round_transmit_param_volume += temp_len
        if self.round_transmit_param_volume == self.flattened_params_length:
            self.restore_queue.put(self.flattened_tenosr_to_restore)
            self.round_transmit_param_volume = 0
            self.round_id += 1

    def update_period(self):
        t1 = time.time()
        update_index = self.connection.root.update_period(self.rank)
        update_index = rpyc.classic.obtain(update_index)
        t2 = time.time()
        self.logging('inner_update_time: %.4f' % (t2-t1))
        self.update_index_queue.put(update_index)
        

class CAS_Interlayer():
    def __init__(self, model, init_period, rank, server_ip, server_port) -> None:
        ''' sync_variable '''
        self.model = model
        self.rank = rank
        self.iter_id = 0
        self.round_id = 0
        self.flattened_params_length = 0
        self.global_period = init_period
        self.init_period = init_period
        self.sync_params_volume = 0

        self.tensor_of_flattened_params = None
        self.freezing_tensor_of_flattened_params = None
        self.received_tensor = None
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.init_model_structure_related_variables()

        self.param_sync_period = torch.full([self.flattened_params_length], self.global_period).cuda() if CUDA \
                                else torch.full([self.flattened_params_length], self.global_period)
        self.mask = torch.zeros(self.flattened_params_length, dtype=torch.bool).cuda() if CUDA \
                    else torch.zeros(self.flattened_params_length, dtype=torch.bool)
        self.freezing_mask = torch.zeros(self.flattened_params_length, dtype=torch.bool).cuda() if CUDA \
                            else torch.zeros(self.flattened_params_length, dtype=torch.bool)

        self.period_list = [64, 32, 16, 8, 4, 2, 1]

        ''' queues(processes common memory) '''
        self.transmit_queue = mp.Queue(10)
        self.period_queue = mp.Queue(10)
        self.restore_queue = mp.Queue(10)
        self.mask_queue = mp.Queue(10)
        self.update_period_tag = mp.Queue(1)
        self.update_index_queue = mp.Queue(1)

        ''' create sync process '''
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for param in self.model.parameters():
            flattened_param = param.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param), 0)
        if CUDA: self.tensor_of_flattened_params = self.tensor_of_flattened_params.cpu()
        self.transmit_queue.put(self.tensor_of_flattened_params)

        self.sync_process = Sync_Process(server_ip, server_port, self.rank, init_period, self.transmit_queue, self.period_queue, \
                                         self.restore_queue, self.mask_queue, self.update_period_tag, self.update_index_queue)
        self.sync_process.start()
        self.received_tensor = self.restore_queue.get().cuda() if CUDA else self.restore_queue.get()
        self.restore_model_from_tensor_received()


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
        self.flattened_params_length = len(self.tensor_of_flattened_params)
        self.logging("Param_length: "+str(self.flattened_params_length))
        self.freezing_tensor_of_flattened_params = copy.deepcopy(self.tensor_of_flattened_params)
        self.received_tensor = copy.deepcopy(self.tensor_of_flattened_params)

    def generate_flattened_tensor_to_transmit(self, iter_id):
        """"""
        round_iter = iter_id % self.global_period
        if round_iter == 0: 
            round_iter = self.global_period
            self.round_id += 1
        self.iter_id = iter_id
        if round_iter == 1: 
            self.freezing_mask.fill_(0)
        if len(self.freezing_mask.nonzero()) != 0:
            self.freezing()
        if round_iter not in self.period_list:
            return 0
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        self.mask = (self.param_sync_period == round_iter).cuda() if CUDA else (self.param_sync_period == round_iter)
        length = len(self.mask.nonzero())
        if length == 0: return 0

        self.sync_params_volume += length/1000000
        if self.iter_id % max(self.init_period, 64) == 0: self.logging('iter_id: %d; round: %d; sum_num_of_param: %.4f' % (self.iter_id, self.round_id, self.sync_params_volume))
        for param in self.model.parameters():
            flattened_param = param.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param), 0)
        self.freezing_tensor_of_flattened_params[self.mask] = self.tensor_of_flattened_params[self.mask]
        self.freezing_mask += self.mask


        self.mask_queue.put(self.mask.cpu()) if CUDA else self.mask_queue.put(self.mask)
        self.period_queue.put(round_iter)
        sync_param = self.tensor_of_flattened_params[self.mask].cpu() if CUDA else self.tensor_of_flattened_params[self.mask]
        t1 = time.time()
        self.transmit_queue.put(self.tensor_of_flattened_params[self.mask].cpu()) if CUDA \
            else self.transmit_queue.put(self.tensor_of_flattened_params[self.mask])
        t2 = time.time()
        self.logging('put_time: %.4f' % (t2-t1))
        if round_iter == self.global_period:
            self.received_tensor = self.restore_queue.get().cuda() if CUDA else self.restore_queue.get()
            t3 = time.time()
            self.logging('outer_sync_time: %.4f' % (t3-t1))
            return 1
        else:
            return 0
        

        
    def update_period(self):
        self.update_period_tag.put(1)
        index = self.update_index_queue.get()
        global_tune_tag = index[-1:]
        index = index[:-1].cuda() if CUDA else index[:-1]
        self.param_sync_period[index] //= 2
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
        self.global_period //= 2
        self.param_sync_period[mask] //= 2

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
        self.transmit_interlayer = CAS_Interlayer(self.model, self.init_period, self.rank, sync_profile['server_ip'], sync_profile['server_port'])


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
        if iter_id % self.init_period == 0:
            t1 = time.time()
            self.transmit_interlayer.update_period()
            t2 = time.time()
            self.logging('update_period_time: %.4f' % (t2-t1))
        self.transmit_interlayer.restore_model_from_tensor_received()
        return True