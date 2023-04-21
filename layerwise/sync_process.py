import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

import threading, queue

torch.set_printoptions(precision=20)
CUDA = torch.cuda.is_available()

tensor_list_to_restore = None
round_transmit_layer_num = 0
period_queue = queue.Queue(10)
current_sync_period = 0
update_index = None
sync_lock = threading.Lock()

def sync_thread(rank, round_id, tensor_list_to_transmit, sync_layer_num, temp_period):
        global current_sync_period, period_queue, sync_lock
        while current_sync_period != temp_period:
            sync_lock.acquire()
            if current_sync_period == 0: 
                 current_sync_period = period_queue.get()
            sync_lock.release()
            time.sleep(0.01)

        logging('start_sync: %d' % temp_period)
        t1 = time.time()
        tensor_list_received = connection.root.aggregate(rank, round_id, tensor_list_to_transmit, temp_period)
        t2 = time.time()
        tensor_list_received = rpyc.classic.obtain(tensor_list_received)
        t3 = time.time()
        logging('temp_period %d inner_sync_time: %.4f, obtain_time: %.4f' % (temp_period,t3-t1,t3-t2))
        current_sync_period = 0
        global tensor_list_to_restore
        j = 0
        for i in sync_layer_num:
             tensor_list_to_restore[i] = tensor_list_received[j]
             j += 1
        global round_transmit_layer_num
        round_transmit_layer_num += len(sync_layer_num)

def sync_and_update_thread(rank, round_id, tensor_list_to_transmit, sync_layer_num, temp_period):
        global current_sync_period, period_queue, sync_lock
        while current_sync_period != temp_period:
            sync_lock.acquire()
            if current_sync_period == 0: 
                current_sync_period = period_queue.get()
            sync_lock.release()
            time.sleep(0.01)
        
        logging('start_sync: %d' % temp_period)
        t1 = time.time()
        tensor_list_received = connection.root.aggregate_and_update(rank, round_id, tensor_list_to_transmit, temp_period)
        t2 = time.time()
        tensor_list_received = rpyc.classic.obtain(tensor_list_received)
        t3 = time.time()
        logging('temp_period %d inner_sync_time: %.4f, obtain_time: %.4f' % (temp_period ,t3-t1,t3-t2))
        current_sync_period = 0
        global tensor_list_to_restore
        j = 0
        for i in sync_layer_num:
             tensor_list_to_restore[i] = tensor_list_received[j]
             j += 1
        global update_index
        update_index = tensor_list_received[-1]
        global round_transmit_layer_num
        round_transmit_layer_num += len(sync_layer_num)

def logging(string):
    print('['+str(datetime.datetime.now())+'] [Sync Process] '+str(string))
    sys.stdout.flush()

class Sync_Service(Service):
    def __init__(self, server_ip, server_port, rank):
        self.rank = rank
        self.round_id = 0
        self.flattened_params_length = 0
        self.init_period = 0
        self.layer_num = 0


        ''' connect to the FL server '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        global connection 
        connection = rpyc.connect(server_ip, server_port, config=rpyc_config)
        connection._config['sync_request_timeout']=120

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Sync Process] '+str(string))
        sys.stdout.flush()

    def exposed_sync_initial_model(self, tensor_list_to_transmit, init_period):
        self.init_period = init_period
        tensor_list_to_transmit = [rpyc.classic.obtain(i) for i in tensor_list_to_transmit]
        self.layer_num = len(tensor_list_to_transmit)
        tensor_list_received = connection.root.sync_initial_model(self.rank, tensor_list_to_transmit)
        tensor_list_received = [rpyc.classic.obtain(i) for i in tensor_list_received]
        global tensor_list_to_restore
        tensor_list_to_restore = copy.deepcopy(tensor_list_received)
        self.round_id += 1
        return tensor_list_to_restore


    def exposed_sync(self, tensor_list_to_transmit, sync_layer_id, temp_period):
        global period_queue
        period_queue.put(temp_period)
        # self.logging('sync_start: %d' % temp_period)
        ''' aggregate '''
        tensor_list_to_transmit = [rpyc.classic.obtain(i) for i in tensor_list_to_transmit]
        sync_layer_id = rpyc.classic.obtain(sync_layer_id)
        t1 = time.time()
        self.logging('create_sync_thread: %d' % temp_period)
        temp_sync_thread = threading.Thread(target=sync_thread, args=(self.rank, self.round_id, tensor_list_to_transmit, sync_layer_id, temp_period,))
        temp_sync_thread.start()
        t2 = time.time()
        self.logging('create_finishing: %.4f' % (t2-t1))
        ''' restore '''
        # flattened_tensor_to_restore[mask] = tensor_to_restore

    def exposed_sync_and_update(self, tensor_list_to_transmit, sync_layer_id, temp_period):
        global period_queue
        period_queue.put(temp_period)
        # self.logging('sync_start: %d' % temp_period)
        ''' aggregate '''
        tensor_list_to_transmit = [rpyc.classic.obtain(i) for i in tensor_list_to_transmit]
        sync_layer_id = rpyc.classic.obtain(sync_layer_id)
        t1 = time.time()
        self.logging('create_sync_thread: %d' % temp_period)
        temp_sync_thread = threading.Thread(target=sync_and_update_thread, args=(self.rank, self.round_id, tensor_list_to_transmit, sync_layer_id, temp_period,))
        temp_sync_thread.start()
        t2 = time.time()
        self.logging('create_finishing: %.4f' % (t2-t1))
        ''' restore '''
        # flattened_tensor_to_restore[mask] = tensor_to_restore

    def exposed_get_restore_tensor(self):
        self.logging('get_restore_start')
        global round_transmit_layer_num
        while round_transmit_layer_num != self.layer_num:
            time.sleep(0.01)
        round_transmit_layer_num = 0
        self.round_id += 1
        self.logging('get_resotre_end')
        global tensor_list_to_restore
        return tensor_list_to_restore

    def exposed_update_period(self):
        global update_index
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