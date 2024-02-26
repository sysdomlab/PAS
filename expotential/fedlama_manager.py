import torch
import rpyc
import copy, sys, numpy, math, datetime, pprint, time, os
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist


CUDA = torch.cuda.is_available()


class FedLAMA_Interlayer:
    def __init__(self, model, init_period):
        self.model = model
        self.init_period = init_period
        self.interlayer_type = 'FedLAMA'

        self.param_shape_list = []
        self.cutoff_index_list = []

        start_index = 0
        layer_num = 0
        for param in self.model.parameters():
            self.param_shape_list.append(param.data.shape)
            end_index = start_index + len(param.data.view(-1))
            self.cutoff_index_list.append([start_index, end_index])
            start_index = end_index
            layer_num += 1

        self.layer_num = layer_num
        self.next_sync_iter_l = [init_period] * layer_num
    
    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] [FedLAMA Interlayer] '+str(string))
        sys.stdout.flush()
    
    def generate_tensor_list_to_transmit(self, iter_id):
        layer_list_to_transmit = []
        tensor_list_to_transmit = []
        for layer_id in range(self.layer_num):
            if iter_id == self.next_sync_iter_l[layer_id]:
                layer_list_to_transmit.append(layer_id)
        if layer_list_to_transmit == []:
            return [], []        
        for i, param in enumerate(self.model.parameters()):
            if i in layer_list_to_transmit:
                tensor_list_to_transmit.append(param.data)
        return layer_list_to_transmit, tensor_list_to_transmit
    
    def restore_model_from_tensor_list_received(self, tensor_list_received, layer_list_received):
        for i, param in enumerate(self.model.parameters()):
            if i in layer_list_received:
                param.data = tensor_list_received[layer_list_received.index(i)]
                

class FedLAMA_Manager():
    """ This object shall be able to run without GIFT (support standard FL by default)"""
    def __init__(self, model, sync_profile) -> None:
        self.model = model
        self.sync_round_id = 0
        self.rank = sync_profile['rank']
        self.init_period = sync_profile['init_period']
        self.interlayer_type = sync_profile['interlayer_type']
        self.transmit_interlayer = FedLAMA_Interlayer(self.model, self.init_period)

        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        rpyc_config['sync_request_timeout'] = 1500
        self.connection = rpyc.connect(sync_profile['server_ip'], sync_profile['server_port'], config=rpyc_config)

        self.sync_init_model()

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] '+str(string))
        sys.stdout.flush()

    def sync_init_model(self):
        tensor_list_to_transmit = []
        for i, param in enumerate(self.model.parameters()):
            tensor_list_to_transmit.append(param.data)
        tensor_list_to_transmit = [t.cpu() for t in tensor_list_to_transmit]
        tensor_list_received = self.connection.root.sync_initial_model(self.rank, tensor_list_to_transmit)
        tensor_list_received = rpyc.classic.obtain(tensor_list_received)
        if CUDA:
            tensor_list_received = [tensor.cuda() for tensor in tensor_list_received]
        self.transmit_interlayer.restore_model_from_tensor_list_received(tensor_list_received, range(len(tensor_list_received)))

    def try_sync_model(self, iter_id, loss_list):
        """ Conduct remote synchronization. """
        t1 = time.time()
        layer_list_to_transmit, tensor_list_to_transmit = self.transmit_interlayer.generate_tensor_list_to_transmit(iter_id)
        if len(tensor_list_to_transmit) != 0:
            t2 = time.time()
            self.logging('generate_time: %.4f' % (t2-t1))
            self.sync_round_id += 1
            tensor_list_to_transmit = [i.cpu() for i in tensor_list_to_transmit] if CUDA else tensor_list_to_transmit
            t1 = time.time()
            tensor_list_received, next_sync_iter_l = self.connection.root.aggregate(self.rank, iter_id, tensor_list_to_transmit, layer_list_to_transmit)
            t2 = time.time()
            tensor_list_received = rpyc.classic.obtain(tensor_list_received)
            self.transmit_interlayer.next_sync_iter_l = rpyc.classic.obtain(next_sync_iter_l)
            t3 = time.time()
            self.logging('sync_time: %.4f, obtain_time: %.4f' % (t3-t1,t3-t2))
            self.logging("receive global model layers: %s" % (layer_list_to_transmit))
            self.logging("next_sync_iter_l: %s" % (next_sync_iter_l))
            if CUDA:
                tensor_list_received = [tensor.cuda() for tensor in tensor_list_received]
            self.transmit_interlayer.restore_model_from_tensor_list_received(tensor_list_received, layer_list_to_transmit)
            return True
        return False