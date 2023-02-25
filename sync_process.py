import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

CUDA = torch.cuda.is_available()

class Sync_Service(Service):
    def __init__(self, server_ip, server_port, rank):
        self.rank = rank
        self.round_id = 0
        self.flattened_params_length = 0
        self.flattened_tensor_to_restore = None
        self.init_period = 0
        self.round_transmit_param_volume = 0

        ''' connect to the FL server '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect(server_ip, server_port, config=rpyc_config)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Sync Process] '+str(string))
        sys.stdout.flush()

    def exposed_sync_initial_model(self, tensor_to_transmit, init_period):
        self.init_period = init_period
        tensor_to_transmit = rpyc.classic.obtain(tensor_to_transmit)
        self.flattened_params_length = len(tensor_to_transmit)
        tensor_to_restore = self.connection.root.sync_initial_model(self.rank, self.init_period, tensor_to_transmit)
        tensor_to_restore = rpyc.classic.obtain(tensor_to_restore)
        self.flattened_tensor_to_restore = copy.deepcopy(tensor_to_restore)
        self.mask = torch.zeros(self.flattened_params_length, dtype=torch.bool)
        self.round_id += 1
        return tensor_to_restore

    def exposed_sync(self, transmit_list, temp_period):
        ''' aggregate '''
        # tensor_to_transmit = rpyc.classic.obtain(tensor_to_transmit)
        # self.mask = rpyc.classic.obtain(mask)
        transmit_list = rpyc.classic.obtain(transmit_list)
        tensor_to_transmit = transmit_list[0]
        mask = transmit_list[1]
        temp_len = len(tensor_to_transmit)
        # t1 = time.time()
        tensor_to_restore = self.connection.root.aggregate(self.rank, self.round_id, tensor_to_transmit, temp_period)
        tensor_to_restore = rpyc.classic.obtain(tensor_to_restore)
        # t2 = time.time()
        # self.logging('inner_sync_time: %.4f' % (t2-t1))
        ''' restore '''
        self.flattened_tensor_to_restore[mask] = tensor_to_restore
        self.round_transmit_param_volume += temp_len

    def exposed_get_restore_tensor(self):
        while self.round_transmit_param_volume != self.flattened_params_length:
            time.sleep(0.01)
        self.round_transmit_param_volume = 0
        self.round_id += 1
        return self.flattened_tensor_to_restore

    def exposed_update_period(self):
        # t1 = time.time()
        update_index = self.connection.root.update_period(self.rank)
        update_index = rpyc.classic.obtain(update_index)
        # t2 = time.time()
        # self.logging('inner_update_time: %.4f' % (t2-t1))
        return update_index
        


if __name__ == "__main__":

    ''' Parse arguments and create APF profile. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=int, default=20000)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')

    args = parser.parse_args()
    print('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks + '\n')

    service = Sync_Service(args.server_ip, args.server_port, args.rank)
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    sync_manager = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    sync_manager.start()