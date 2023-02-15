import torch
import copy, sys, numpy, math, datetime, pprint, time
import queue
import torch.multiprocessing as mp
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist

print(mp.get_all_start_methods())
mp.set_start_method('forkserver')
CUDA = torch.cuda.is_available()

ParamGIFTProfile = {
    'ema_theta': 0.9,
    'observe_window_size': 5,
    'total_group_size': int(math.log2(128))+2,
}


class Sync_Process(mp.Process):
    def __init__(self, world_size, rank, init_method, gather_queue, restore_queue, mask_queue, grad_consistency_queue, tag_queue, index_queue, global_tune_tag) -> None:
        super(Sync_Process, self).__init__()
        self.world_size = world_size
        self.rank = rank
        self.init_method = init_method
        self.flattened_params_length = 0
        self.gather_queue = gather_queue
        self.restore_queue = restore_queue
        self.mask_queue = mask_queue
        self.grad_consistency_queue = grad_consistency_queue
        self.tag_queue = tag_queue
        self.index_queue = index_queue
        self.global_tune_tag = global_tune_tag

        self.sum = 0

        self.restore_tensor = None
        self.ema_theta = ParamGIFTProfile['ema_theta']
        self.positive_grad_ema = torch.zeros(self.flattened_params_length)
        self.negative_grad_ema = torch.zeros(self.flattened_params_length)
        self.init_dist()
        self.logging('create Sync_Process')



    def run(self):
        while True:
            if not self.gather_queue.empty():
                self.sync()
            if self.sum == self.flattened_params_length:
                self.restore_queue.put(self.restore_tensor)
                self.sum = 0
            if not self.tag_queue.empty():
                self.tag_queue.get()  
                if self.rank == 0:
                    self.calculate()  
                self.frequency_tune()
                self.tune_global_frequency()
            time.sleep(0.0001)

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync_Manager] [Sync_Process] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
        sys.stdout.flush()

    def init_dist(self):
        dist.init_process_group(backend='gloo', 
                        init_method=self.init_method,
                        world_size=self.world_size,
                        rank=self.rank)
        self.group = dist.new_group([ i for i in range(self.world_size)])
        self.logging('init_dist_finish')
        while(self.gather_queue.empty()):
            self.logging('empty')
        self.logging(self.gather_queue.empty())
        sync_param = self.gather_queue.get()
        self.logging(self.gather_queue.empty())
        dist.broadcast(sync_param, src=0, group=self.group)
        self.logging('3')
        self.flattened_params_length = len(sync_param)
        self.restore_tensor = copy.deepcopy(sync_param)
        self.restore_queue.put(sync_param)

    def sync(self):
        sync_param = self.gather_queue.get()
        mask = self.mask_queue.get()
        temp_len = len(sync_param)
        if self.rank == 0:
            sync_param_list = [torch.tensor([0.0]*temp_len) for _ in range(self.world_size)]
            dist.gather(sync_param, gather_list = sync_param_list, dst=0)
        else:
            dist.gather(sync_param, dst=0)
        if self.rank == 0:
            sync_param = sum(sync_param_list) / self.world_size
            self.pooling(sync_param_list, mask)
        dist.broadcast(sync_param, src=0, group=self.group)
        self.restore_tensor[mask] = sync_param
        self.sum += temp_len

    def pooling(self, sync_param_list, mask):
        length = sync_param_list[0]
        positive_grad = torch.zeros(length)
        negative_grad = torch.zeros(length)
        ''' pooling '''
        for tensor_received in sync_param_list:
            if CUDA:
                tensor_received = tensor_received
            grad_tensor = tensor_received - self.restore_tensor[mask]
            positive_grad += torch.where(grad_tensor >= 0, grad_tensor, 0.0)
            negative_grad -= torch.where(grad_tensor < 0, grad_tensor, 0.0)

        ''' smoothing '''
        self.positive_grad_ema[mask] = self.ema_theta*self.positive_grad_ema[mask] + (1 - self.ema_theta)*positive_grad
        self.negative_grad_ema[mask] = self.ema_theta*self.negative_grad_ema[mask] + (1 - self.ema_theta)*negative_grad
    
    def calculate(self):
        grad_consistency = abs(self.positive_grad_ema-self.negative_grad_ema) / (self.positive_grad_ema+self.negative_grad_ema)
        self.grad_consistency_queue.put(grad_consistency)

    def frequency_tune(self):
        length = torch.tensor([0])
        if self.rank == 0: 
            index1 = self.index_queue.get()
            length = torch.tensor([len(index1)])
        dist.broadcast(length, src=0, group=self.group)
        if self.rank == 0:
            if length != 0:
                dist.broadcast(index1, src=0, group=self.group)
        else:
            if length != 0:
                index1 = torch.zeros([length[0],1],dtype=torch.long)
                dist.broadcast(index1, src=0, group=self.group)
            else:
                index1 = torch.zeros([], dtype=torch.int64)
            self.index_queue.put(index1)
    
    def tune_global_frequency(self):
        if self.rank == 0: 
            tune = self.global_tune_tag.get()
            dist.broadcast(tune, src=0)
        else: 
            tune = torch.tensor([0])
            dist.broadcast(tune, src=0)
            self.global_tune_tag.put(tune)



class ParamGIFT_Interlayer():
    def __init__(self, model, sync_frequency, hyperparams, rank, world_size, init_method) -> None:
        self.interlayer_type = 'ParamGIFT'
        self.model = model
        self.sync_frequency = sync_frequency
        self.rank = rank
        self.world_size = world_size
        self.init_method = init_method
        self.iter_id = 0
        self.round = 0
        self.sum_num_of_param = 0.0
        self.frequency_list = []
        i = 1
        while i <= sync_frequency:
            self.frequency_list.append(i)
            i *= 2
        print(self.frequency_list)

        """ Set hyper_parameters """
        self.ema_theta = hyperparams['ema_theta']

        """ Initialize model-structure-related variables """
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.tensor_of_flattened_params = None
        self.frozen_tensor_of_flattened_params = None
        self.received_tensor = None
        self.flattened_params_length = -1
        self.init_model_structure_related_variables()


        """  Initialize GIFT-related variables """
        self.mask = torch.zeros(self.flattened_params_length).cuda().bool() if CUDA else torch.zeros(self.flattened_params_length).bool() # 旧版本不支持bool
        self.freezing_mask = torch.zeros(self.flattened_params_length).cuda().bool() if CUDA else torch.zeros(self.flattened_params_length).bool()
        # self.param_sync_iter = torch.full([self.flattened_params_length], sync_frequency).cuda() if CUDA else torch.full([self.flattened_params_length], sync_frequency)
        self.param_sync_frequency = torch.full([self.flattened_params_length], sync_frequency).cuda() if CUDA else torch.full([self.flattened_params_length], sync_frequency)
        # self.next_sync_iter = sync_frequency
        self.max_frequency = sync_frequency
        self.gather_queue = mp.Queue(10)
        self.restore_queue = mp.Queue(10)
        self.mask_queue = mp.Queue(10)
        self.grad_consistency_queue = mp.Queue(1)
        self.tag_queue = mp.Queue(1)
        self.index_queue = mp.Queue(1)
        self.global_tune_tag = mp.Queue(1)
        if self.rank == 0:
            self.max_observe_window_start = 0
            self.min_average_consistency = 1
            self.min_average_start_round = 0
            self.observe_window_size = torch.full([self.flattened_params_length], hyperparams['observe_window_size']).cuda() if CUDA else torch.full([self.flattened_params_length], hyperparams['observe_window_size'])
            self.average_consistency_stable_round = self.observe_window_size
            self.grad_consistency = torch.ones(self.flattened_params_length).cuda() if CUDA else torch.ones(self.flattened_params_length)
            self.min_grad_consistency = torch.ones(self.flattened_params_length).cuda() if CUDA else torch.ones(self.flattened_params_length)
            self.positive_grad_ema = torch.zeros(self.flattened_params_length).cuda() if CUDA else torch.zeros(self.flattened_params_length)
            self.negative_grad_ema = torch.zeros(self.flattened_params_length).cuda() if CUDA else torch.zeros(self.flattened_params_length)
            self.window_start_round = torch.zeros(self.flattened_params_length).cuda() if CUDA else torch.zeros(self.flattened_params_length)
            self.temp_mask1 = copy.deepcopy(self.mask)
            self.temp_mask2 = copy.deepcopy(self.mask)
        self.logging('1')
        self.gather_queue.put(self.tensor_of_flattened_params)
        self.sync_process = Sync_Process(self.world_size, self.rank, self.init_method, self.gather_queue, self.restore_queue, self.mask_queue, self.grad_consistency_queue, self.tag_queue, self.index_queue, self.global_tune_tag)
        self.logging('creat_process_finish')
        self.received_tensor = self.restore_queue.get().cuda() if CUDA else self.restore_queue.get()
        self.restore_model_from_tensor_received()
        self.sync_process.start()
        self.logging('create CAS Interlayer')



    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync_Manager] [ParamGIFT_Interlayer] '+str(string))
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
        self.frozen_tensor_of_flattened_params = copy.deepcopy(self.tensor_of_flattened_params)
        self.received_tensor = copy.deepcopy(self.tensor_of_flattened_params)
        if self.rank == 0:
            self.tensor_list_received = [torch.zeros(self.flattened_params_length) for _ in range(self.world_size)]
            
    def generate_flattened_tensor_to_transmit(self, iter_id):
        """"""
        round_iter = iter_id % self.max_frequency
        if round_iter == 0: 
            round_iter = self.max_frequency
            self.round += 1
        self.iter_id = iter_id
        if round_iter == 1: 
            self.freezing_mask.fill_(0)
        if len(self.freezing_mask.nonzero()) != 0:
            self.freezing()
        if round_iter not in self.frequency_list:
            return 0
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        self.mask = (self.param_sync_frequency == round_iter).cuda() if CUDA else (self.param_sync_frequency == round_iter)
        length = len(self.mask.nonzero())
        if length == 0: return 0

        self.sum_num_of_param += length/1000000
        for param in self.model.parameters():
            flattened_param = param.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param), 0)
        self.frozen_tensor_of_flattened_params[self.mask] = self.tensor_of_flattened_params[self.mask]
        self.freezing_mask += self.mask
        self.mask_queue.put(self.mask)
        self.gather_queue.put(self.tensor_of_flattened_params[self.mask].cpu()) if CUDA else self.gather_queue.put(self.tensor_of_flattened_params[self.mask])
        if round_iter == self.max_frequency:
            self.received_tensor = self.restore_queue.get().cuda() if CUDA else self.restore_queue.get()
            return 1
        else:
            return 0
        

    def calculate_grad_consistency(self):
        self.grad_consistency = self.grad_consistency_queue.get()
        self.temp_mask1 = ((self.grad_consistency > self.min_grad_consistency) \
                            & (self.grad_consistency < 0.3) & (self.param_sync_frequency > 1) \
                            & (self.round > self.window_start_round + self.observe_window_size)).cuda() \
                            if CUDA else ((self.grad_consistency > self.min_grad_consistency) \
                            & (self.grad_consistency < 0.3) & (self.param_sync_frequency > 1) \
                            & (self.round > self.window_start_round + self.observe_window_size))
        self.temp_mask2 = (self.grad_consistency <= self.min_grad_consistency).cuda() \
                            if CUDA else (self.grad_consistency <= self.min_grad_consistency)
        # self.temp_mask3 = ((self.grad_consistency >= 0.5) & (self.param_sync_frequency < self.max_frequency) & (self.min_grad_consistency < 0.3))
        # self.window_start_iter[self.temp_mask3] = self.iter_id
        self.param_sync_frequency[self.temp_mask1] //= 2
        # self.temp_mask4 = self.temp_mask1 + self.temp_mask3
        self.min_grad_consistency[self.temp_mask1] = 1
        self.observe_window_size[self.temp_mask1] *= 2
        self.negative_grad_ema[self.temp_mask1] = 0
        self.positive_grad_ema[self.temp_mask1] = 0
        self.window_start_round[self.temp_mask1] = self.round
        self.window_start_round[self.temp_mask2] = self.round
        self.min_grad_consistency[self.temp_mask2] = self.grad_consistency[self.temp_mask2]
        # self.param_sync_frequency[self.temp_mask3] *= 2
        # self.min_grad_consistency[self.temp_mask3] = 1
        # self.negative_grad_ema[self.temp_mask3] = 0
        # self.positive_grad_ema[self.temp_mask3] = 0
        
        return self.temp_mask1.nonzero().cpu()
        
    def update_frequency(self, index1):
        if self.rank != 0: 
            self.param_sync_frequency[index1] //= 2
            # self.param_sync_frequency[index2] *= 2

    def freezing(self):
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for param in self.model.parameters():
            flattened_param = param.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param), 0)
        self.tensor_of_flattened_params[self.freezing_mask] = self.frozen_tensor_of_flattened_params[self.freezing_mask]
        for i, param in enumerate(self.model.parameters()):
            param.data = self.tensor_of_flattened_params[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])


    def tune_max_frequency(self):
        if self.rank == 0:
            j = self.sync_frequency
            ll = []
            gc_l = []
            while j >= 1:
                mask = (self.param_sync_frequency == j)
                index = mask.nonzero()
                l=len(index)
                if j == self.max_frequency and l == 0:
                    tune = torch.tensor([1])
                if l != 0:
                    proportion = l/self.flattened_params_length
                    ll.append(round(proportion*100,3))
                    gc = self.grad_consistency[mask].sum() / l
                    # waste += (1 - j/self.max_frequency) * proportion
                    if j == self.max_frequency: 
                        if gc < 0.9:
                            self.min_average_start_round = max(self.round,self.min_average_start_round)
                        if proportion < 0.01:
                            tune = torch.tensor([1])
                else:
                    ll.append(0)
                    gc= 0
                gc_l.append(round(float(gc),4))
                j //= 2
            self.logging("sync_frequency_distribution: "+str(ll))
            self.logging("param_averaging_grad_consistency: "+str(gc_l))
            self.logging('iter_id: %d; round: %d; sum_num_of_param: %.4f' % (self.iter_id, self.round, self.sum_num_of_param))
            average_consistency = self.grad_consistency.sum() / self.flattened_params_length  
            self.logging("iter_id: %d; average_consistency: %.4f" % (self.iter_id, average_consistency))

            """print different parameters' consistency and frequency"""
            # index = torch.tensor([10000, 20000, 30000, 40000, 50000])
            # some_parameters_consistency =  self.grad_consistency[index].reshape(-1).tolist()
            # some_parameters_consistency = [round(i, 4) for i in some_parameters_consistency]
            # some_parameters_frequency = self.param_sync_frequency[index].reshape(-1).int().tolist()
            # self.logging("some_parameters_consistency: "+str(some_parameters_consistency))
            # self.logging("some_parameters_frequency: "+str(some_parameters_frequency))

            # if average_consistency <= self.min_average_consistency or (average_consistency - self.min_average_consistency > 0.005):
            #     self.min_average_consistency = average_consistency
            #     self.min_average_start_round = self.round
            # else:
            #     if self.min_average_start_round + self.observe_window_size * 2 < self.round and self.max_frequency > 1:
            #         tune = torch.tensor([1])

            
            
            # if self.average_consistency_stable_round + self.observe_window_size < self.iter_id:
            #     tune = torch.tensor([1])
        
        if self.rank == 0:
            self.global_tune_tag.put(tune)
        else:
            tune = self.global_tune_tag.get()
        if tune == 1:
            mask = (self.param_sync_frequency == self.max_frequency)
            self.logging('tune max frequency')
            self.max_frequency //= 2
            self.param_sync_frequency[mask] //= 2
            if self.rank == 0:
                self.observe_window_size[mask] *= 2
                self.positive_grad_ema[mask] = 0
                self.negative_grad_ema[mask] = 0
                self.window_start_round.fill_(self.round)
                self.min_grad_consistency.fill_(1)
                self.min_average_start_round = self.round

    def restore_model_from_tensor_received(self):
        """"""
        for i, param in enumerate(self.model.parameters()):
            param.data = self.received_tensor[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])


class ParamGIFT_Manager():
    def __init__(self, model, sync_profile) -> None:
        self.model = model
        self.sync_round_id = 0
        self.sync_frequency = sync_profile['sync_frequency']
        self.interlayer_type = sync_profile['interlayer_type']
        self.dist_profile = sync_profile['dist_profile']
        self.rank = self.dist_profile['rank']
        self.world_size = self.dist_profile['world_size']
        self.transmit_interlayer = ParamGIFT_Interlayer(self.model, self.sync_frequency, ParamGIFTProfile, self.rank, self.world_size, self.dist_profile['master_address'])

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [ParamGIFT Manager] '+str(string))
        sys.stdout.flush()
        

    def try_sync_model(self, iter_id):
        """ Conduct remote synchronization. """
        if self.transmit_interlayer.generate_flattened_tensor_to_transmit(iter_id) == 0:
            return False
        self.sync_round_id += 1
        ''' [2]average '''
        if iter_id % 64 == 0:
            self.transmit_interlayer.tag_queue.put(1)
            if self.rank == 0:
                index1= self.transmit_interlayer.calculate_grad_consistency()
                self.transmit_interlayer.index_queue.put(index1)
            else:
                tune_index = self.transmit_interlayer.index_queue.get().cuda() if CUDA else self.transmit_interlayer.index_queue.get()
                self.transmit_interlayer.update_frequency(tune_index)
            self.transmit_interlayer.tune_max_frequency()

        ''' [3]broadcast '''
        # t3 = time.time()
        # print("broadcast_time: %.4f" % (t4-t3))
        self.transmit_interlayer.restore_model_from_tensor_received()
        # t5 = time.time()
        # print("restore_time: %.4f" % (t5-t4))

        return True