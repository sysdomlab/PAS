import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

import threading, queue

CUDA = torch.cuda.is_available()

flattened_tensor_to_restore = None
round_transmit_param_volume = 0
period_queue = queue.Queue(10)
current_sync_period = 0
update_index = None

def sync_thread(rank, round_id, tensor_to_transmit, mask, temp_period):
        global current_sync_period
        global period_queue
        while current_sync_period != temp_period:
            if current_sync_period == 0: current_sync_period = period_queue.get()
            time.sleep(0.01)
        logging('start_sync: %d' % temp_period)
        t1 = time.time()
        tensor_to_restore = connection.root.aggregate(rank, round_id, tensor_to_transmit, temp_period)
        tensor_to_restore = rpyc.classic.obtain(tensor_to_restore)
        t2 = time.time()
        logging('inner_sync_time: %.4f' % (t2-t1))
        current_sync_period = 0
        global flattened_tensor_to_restore
        flattened_tensor_to_restore[mask] = tensor_to_restore
        temp_len = len(tensor_to_transmit)
        global round_transmit_param_volume
        round_transmit_param_volume += temp_len

def sync_and_update_thread(rank, round_id, tensor_to_transmit, mask, temp_period):
        global current_sync_period
        global period_queue
        while current_sync_period != temp_period:
            if current_sync_period == 0: current_sync_period = period_queue.get()
            time.sleep(0.01)
        logging('start_sync: %d' % temp_period)
        t1 = time.time()
        temp_list = connection.root.aggregate_and_update(rank, round_id, tensor_to_transmit, temp_period)
        temp_list = rpyc.classic.obtain(temp_list)
        t2 = time.time()
        logging('inner_sync_time: %.4f' % (t2-t1))
        current_sync_period = 0
        global flattened_tensor_to_restore
        flattened_tensor_to_restore[mask] = temp_list[0]
        global update_index
        update_index = temp_list[1]
        logging(update_index)
        temp_len = len(tensor_to_transmit)
        global round_transmit_param_volume
        round_transmit_param_volume += temp_len

def logging(string):
    print('['+str(datetime.datetime.now())+'] [Sync Process] '+str(string))
    sys.stdout.flush()

class Sync_Service(Service):
    def __init__(self, server_ip, server_port, rank):
        self.rank = rank
        self.round_id = 0
        self.flattened_params_length = 0
        self.init_period = 0


        ''' connect to the FL server '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        global connection 
        connection = rpyc.connect(server_ip, server_port, config=rpyc_config)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Sync Process] '+str(string))
        sys.stdout.flush()

    def exposed_sync_initial_model(self, tensor_to_transmit, init_period):
        self.init_period = init_period
        tensor_to_transmit = rpyc.classic.obtain(tensor_to_transmit)
        self.flattened_params_length = len(tensor_to_transmit)
        tensor_to_restore = connection.root.sync_initial_model(self.rank, self.init_period, tensor_to_transmit)
        tensor_to_restore = rpyc.classic.obtain(tensor_to_restore)
        global flattened_tensor_to_restore
        flattened_tensor_to_restore = copy.deepcopy(tensor_to_restore)
        self.mask = torch.zeros(self.flattened_params_length, dtype=torch.bool)
        self.round_id += 1
        return tensor_to_restore


    def exposed_sync(self, tensor_to_transmit, mask, temp_period):
        global period_queue
        period_queue.put(temp_period)
        self.logging('sync_start: %d' % temp_period)
        ''' aggregate '''
        tensor_to_transmit = rpyc.classic.obtain(tensor_to_transmit)
        mask = rpyc.classic.obtain(mask)
        t1 = time.time()
        self.logging('create_sync_thread: %d' % temp_period)
        temp_sync_thread = threading.Thread(target=sync_thread, args=(self.rank, self.round_id, tensor_to_transmit, mask, temp_period,))
        temp_sync_thread.start()
        t2 = time.time()
        self.logging('create_finishing: %.4f' % (t2-t1))
        ''' restore '''
        # flattened_tensor_to_restore[mask] = tensor_to_restore

    def exposed_sync_and_update(self, tensor_to_transmit, mask, temp_period):
        global period_queue
        period_queue.put(temp_period)
        self.logging('sync_start: %d' % temp_period)
        ''' aggregate '''
        tensor_to_transmit = rpyc.classic.obtain(tensor_to_transmit)
        mask = rpyc.classic.obtain(mask)
        t1 = time.time()
        self.logging('create_sync_thread: %d' % temp_period)
        temp_sync_thread = threading.Thread(target=sync_and_update_thread, args=(self.rank, self.round_id, tensor_to_transmit, mask, temp_period,))
        temp_sync_thread.start()
        t2 = time.time()
        self.logging('create_finishing: %.4f' % (t2-t1))
        ''' restore '''
        # flattened_tensor_to_restore[mask] = tensor_to_restore

    def exposed_get_restore_tensor(self):
        self.logging('get_restore_start')
        global round_transmit_param_volume
        while round_transmit_param_volume != self.flattened_params_length:
            time.sleep(0.01)
        round_transmit_param_volume = 0
        self.round_id += 1
        self.logging('get_resotre_end')
        global flattened_tensor_to_restore
        return flattened_tensor_to_restore

    def exposed_update_period(self):
        global update_index
        logging(update_index)
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
    sys.stdout.flush()

    service = Sync_Service(args.server_ip, args.server_port, args.rank)
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    sync_manager = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    sync_manager.start()