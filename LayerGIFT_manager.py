import torch
import copy, sys, numpy, math, datetime, pprint, time, math
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist

CUDA = torch.cuda.is_available()

LayerGIFTProfile = {
    'ema_theta': 0.9,
    'observe_window_size': 10,
    'max_frequency_tune_threshold': 0.5
}

class LayerGIFT_Interlayer():
    def __init__(self, model, sync_frequency, hyperparams, rank) -> None:
        self.interlayer_type = 'LayerGIFT'
        self.model = model
        self.sync_frequency = sync_frequency
        self.rank = rank
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
        self.observe_window_size = hyperparams['observe_window_size'] * sync_frequency
        self.max_frequency_tune_threshold = hyperparams['max_frequency_tune_threshold']

        """ Initialize model-structure-related variables """
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.tensor_of_flattened_params = None
        self.last_tensor_of_flattened_params = None
        self.flattened_params_length = -1
        self.layernum = 0
        self.layer_len = []
        self.init_model_structure_related_variables()
        if self.layernum % 2 != 1 :
            self.logging("layernum error")
            return KeyError
        else:
            self.layernum = (self.layernum + 1) // 2
        self.logging(("layer_num: %d, layer_size: " % self.layernum)+ str(self.layer_len))


        """  Initialize GIFT-related variables """
        self.mask = torch.zeros(self.layernum).cuda().byte() if CUDA else torch.zeros(self.layernum).byte() # 旧版本不支持bool
        self.index = None
        self.freezing_mask = torch.zeros(self.layernum).cuda().byte() if CUDA else torch.zeros(self.layernum).byte()
        self.sync_len = 0
        # self.param_sync_iter = torch.full([self.flattened_params_length], sync_frequency).cuda() if CUDA else torch.full([self.flattened_params_length], sync_frequency)
        self.layer_sync_frequency = torch.full([self.layernum], sync_frequency).cuda() if CUDA else torch.full([self.layernum], sync_frequency)
        # self.next_sync_iter = sync_frequency
        self.max_frequency = sync_frequency
        if self.rank == 0:
            self.average_consistency = 0
            self.max_observe_window_start = 0
            self.min_average_consistency = 1
            self.min_average_start_iter = 0
            self.average_consistency_stable_round = self.observe_window_size
            self.grad_consistency = torch.ones(self.layernum).cuda() if CUDA else torch.ones(self.layernum)
            self.min_grad_consistency = torch.ones(self.layernum).cuda() if CUDA else torch.ones(self.layernum)
            self.positive_grad_ema = []
            self.negative_grad_ema = []
            for len in self.layer_len:
                self.positive_grad_ema.append(torch.zeros(len).cuda()) if CUDA else self.positive_grad_ema.append(torch.zeros(len))
                self.negative_grad_ema.append(torch.zeros(len).cuda()) if CUDA else self.negative_grad_ema.append(torch.zeros(len))
            self.window_start_iter = torch.zeros(self.layernum).cuda() if CUDA else torch.zeros(self.layernum)


    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync_Manager] [ParamGIFT_Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
        sys.stdout.flush()

    def init_model_structure_related_variables(self):
        s_index = 0
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for i, param in enumerate(self.model.parameters()):
            self.param_shape_list.append(param.data.shape)
            flatten_params = param.data.view(-1)
            e_index = s_index + len(flatten_params)
            if i % 2 == 0: self.layer_len.append(len(flatten_params))
            else : self.layer_len[i//2] += len(flatten_params)
            self.cutoff_index_list.append([s_index, e_index])
            s_index = e_index
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flatten_params),0)
        self.layernum = i
        self.flattened_params_length = len(self.tensor_of_flattened_params)
        self.logging("Param_length: %d, Layer_num: %d" % (self.flattened_params_length, self.layernum))
        self.last_tensor_of_flattened_params = copy.deepcopy(self.tensor_of_flattened_params)
            
    def generate_flattened_tensor_to_transmit(self, iter_id):
        """"""
        round_iter = iter_id % self.max_frequency
        if round_iter == 0: round_iter = self.max_frequency
        if round_iter not in self.frequency_list:
            return torch.tensor([])
        self.iter_id = iter_id
        if round_iter == 1: 
            self.freezing_mask.fill_(0)
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        self.mask = (self.layer_sync_frequency == round_iter).cuda() if CUDA else (self.layer_sync_frequency == round_iter)
        self.index = self.mask.nonzero()
        self.sync_len = 0
        for i in self.index:
            self.sync_len += self.layer_len[i]
        if self.sync_len == 0: return torch.tensor([])
        self.sum_num_of_param += self.sync_len / 1000000
        self.round += 1
        for i, param in enumerate(self.model.parameters()):
            if (self.mask[i//2] == 1):
                flattened_param = param.data.view(-1)
                self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param), 0)
        # t2 = time.time()
        # print("generate_time: %.4f" % (t2-t1))
        return self.tensor_of_flattened_params
        

    def pooling(self, tensor_list_received):
        positive_grad = torch.zeros(self.sync_len).cuda() if CUDA else torch.zeros(self.sync_len)
        negative_grad = torch.zeros(self.sync_len).cuda() if CUDA else torch.zeros(self.sync_len)
        zeros = torch.zeros(self.sync_len).cuda() if CUDA else torch.zeros(self.sync_len)
        ''' pooling '''
        last_tensor = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for i in self.index:
            last_layer_tensor = self.last_tensor_of_flattened_params[self.cutoff_index_list[2*i][0]: self.cutoff_index_list[2*i+1][1]]
            last_tensor = torch.cat((last_tensor, last_layer_tensor), 0)
        for tensor_received in tensor_list_received:
            if CUDA:
                tensor_received = tensor_received.cuda()
            grad_tensor = tensor_received - last_tensor
            positive_grad += torch.where(grad_tensor >= 0, grad_tensor, zeros)
            negative_grad -= torch.where(grad_tensor < 0, grad_tensor, zeros)
        s_index = 0
        for i in self.index:
            e_index = s_index + self.layer_len[i]
            self.positive_grad_ema[i] = self.ema_theta*self.positive_grad_ema[i] + (1 - self.ema_theta)*positive_grad[s_index: e_index]
            self.negative_grad_ema[i] = self.ema_theta*self.negative_grad_ema[i] + (1 - self.ema_theta)*negative_grad[s_index: e_index]
            s_index = e_index
            


    def calculate_grad_consistency(self):
        frequency_tune_index = []
        self.average_consistency = 0
        for i in range(self.layernum):
            temp_sum = (abs(self.positive_grad_ema[i]-self.negative_grad_ema[i]) / (self.positive_grad_ema[i]+self.negative_grad_ema[i])).sum()
            self.average_consistency += temp_sum
            self.grad_consistency[i] = temp_sum / self.layer_len[i]
            # if self.grad_consistency[i] <= self.min_grad_consistency[i] or (self.grad_consistency[i] - self.min_grad_consistency[i] > 0.005):
            #     self.min_grad_consistency[i] = self.grad_consistency[i]
            #     self.window_start_iter[i] = self.iter_id
            # else:
            #     if self.iter_id > self.window_start_iter[i] + self.observe_window_size and self.layer_sync_frequency[i] > 1 and self.layer_sync_frequency[i] > self.max_frequency // 2: 
            #         self.layer_sync_frequency[i] //= 2
            #         self.window_start_iter[i] = self.iter_id
            #         self.min_average_start_iter = self.iter_id
            #         self.grad_consistency[i] = 1
            #         self.positive_grad_ema[i].fill_(0)
            #         self.negative_grad_ema[i].fill_(0)
            #         frequency_tune_index.append(i)
        return torch.tensor(frequency_tune_index)
        
    def update_frequency(self, index):
        if self.rank != 0: 
            self.layer_sync_frequency[index] //= 2
            # self.param_sync_frequency[index2] *= 2

    def tune_max_frequency(self):
        tune = torch.tensor([0])
        if self.rank == 0:
            self.logging('iter_id: %d; round: %d; sum_num_of_param: %.4f' % (self.iter_id, self.round, self.sum_num_of_param))
            consistency_ls = [round(i,4) for i in self.grad_consistency.tolist()]
            self.logging(("iter_id: %d; layer_consistency: " % self.iter_id) + str(consistency_ls))
            self.average_consistency /= self.flattened_params_length
            self.logging("iter_id: %d; average_consistency: %.4f" % (self.iter_id, self.average_consistency))
            if self.average_consistency <= self.min_average_consistency or (self.average_consistency - self.min_average_consistency > 0.005):
                self.min_average_consistency = self.average_consistency
                self.min_average_start_iter = self.iter_id
            else:
                if self.min_average_start_iter + self.observe_window_size < self.iter_id and self.max_frequency > 1:
                    tune = torch.tensor([1])
            if self.layer_sync_frequency.max() != self.max_frequency:
                tune = torch.tensor([1])

        dist.broadcast(tune, src=0)
        if tune == 1:
            mask = (self.layer_sync_frequency == self.max_frequency)
            self.logging('tune max frequency')
            self.max_frequency //= 2
            self.layer_sync_frequency[mask] = self.max_frequency
            if self.rank == 0:
                for i in mask.nonzero():
                    self.positive_grad_ema[i].fill_(0)
                    self.negative_grad_ema[i].fill_(0)
                self.min_average_consistency = 1
                self.min_average_start_iter = self.iter_id
        
        self.logging("sync_frequency: " + str(self.layer_sync_frequency.tolist()))

    def restore_model_from_tensor_received(self, tensor_received):
        """"""
        self.freezing_mask += self.mask
        s_index = 0
        for i in self.index:
            self.last_tensor_of_flattened_params[self.cutoff_index_list[2*i][0]:self.cutoff_index_list[2*i+1][1]]=tensor_received[s_index: s_index+self.layer_len[i]]
            s_index += self.layer_len[i]
        restore_tensor = copy.deepcopy(self.last_tensor_of_flattened_params)  # 保存参数会改变指针，需要单独复制一份
        for i, param in enumerate(self.model.parameters()):
            if self.freezing_mask[i//2] == 1:
                param.data = restore_tensor[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])


class LayerGIFT_Manager():
    def __init__(self, model, sync_profile) -> None:
        self.model = model
        self.sync_round_id = 0
        self.world_size = -1
        self.rank = -1
        self.group = None
        self.sync_frequency = int(sync_profile['sync_frequency'])
        self.interlayer_type = sync_profile['interlayer_type']
        self.init_dist(sync_profile['dist_profile'])
        self.transmit_interlayer = LayerGIFT_Interlayer(self.model, self.sync_frequency, LayerGIFTProfile, self.rank)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [LayerGIFT Manager] '+str(string))
        sys.stdout.flush()

    def init_dist(self, dist_profile):
        self.world_size = dist_profile['world_size'] # world_size is required in all_reduce averaging.
        self.rank = dist_profile['rank']
        dist.init_process_group(backend='tcp' if CUDA else 'gloo', 
                                init_method=dist_profile['master_address'],
                                world_size=dist_profile['world_size'],
                                rank=dist_profile['rank'])
        self.group = dist.new_group([ i for i in range(self.world_size)])
        self.model = self.model.cpu() if CUDA else self.model
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0, group=self.group)
        self.model = self.model.cuda() if CUDA else self.model

    def try_sync_model(self, iter_id):
        """ Conduct remote synchronization. """
        # t1 = time.time()
        flattened_tensor_to_transmit = self.transmit_interlayer.generate_flattened_tensor_to_transmit(iter_id)
        # t2 = time.time()
        # print("generate time: %.4f" % (t2-t1))
        if len(flattened_tensor_to_transmit) == 0: return False
        self.sync_round_id += 1
        tensor_len = len(flattened_tensor_to_transmit)
        tensor_received = torch.zeros(tensor_len)
        tensor_list_received = [torch.zeros(tensor_len) for _ in range(self.world_size)] if self.rank == 0 else None

        ''' [1]gather '''
        # t1 = time.time()
        if CUDA:
            flattened_tensor_to_transmit = flattened_tensor_to_transmit.cpu()
            
        if self.rank == 0:
            dist.gather(flattened_tensor_to_transmit, gather_list = tensor_list_received, dst=0, group=self.group)
        else:
            dist.gather(flattened_tensor_to_transmit, dst=0, group=self.group) #老版本torch
        # t2 = time.time()
        # print("gather_time: %.4f" % (t2-t1))

        ''' [2]average '''
        if self.rank == 0:
            tensor_received = sum(tensor_list_received) / self.world_size
            self.transmit_interlayer.pooling(tensor_list_received)
            
        if iter_id % self.sync_frequency == 0:
            length = torch.tensor([0])
            if self.rank == 0:
                index = self.transmit_interlayer.calculate_grad_consistency()
                length = torch.tensor([len(index)])
            dist.broadcast(length, src=0, group=self.group)
            if length != 0:
                if self.rank != 0: 
                    index = torch.zeros(length, dtype=torch.long)
                    # index2 = torch.zeros([length[1],1],dtype=torch.long)
                print(index)
                dist.broadcast(index, src=0, group=self.group)
                print(index)
                # dist.broadcast(index2, src=0, group=self.group)
                self.transmit_interlayer.update_frequency(index)
            self.transmit_interlayer.tune_max_frequency()

            # t3 = time.time()
            # print("calculate_time: %.4f" % (t3-t2))

        ''' [3]broadcast '''
        # t3 = time.time()
        dist.broadcast(tensor_received, src=0, group=self.group)
        # t4 = time.time()
        # print("broadcast_time: %.4f" % (t4-t3))
        if CUDA:
            tensor_received = tensor_received.cuda()
        self.transmit_interlayer.restore_model_from_tensor_received(tensor_received)
        # t5 = time.time()
        # print("restore_time: %.4f" % (t5-t4))

        return True