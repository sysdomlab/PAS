import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time, random

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from threading import Lock,Thread

path=os.path.dirname(__file__)
sys.path.append(path+'/..')
from model_manager import Model_Manager
from dataset_manager import Dataset_Manager

torch.set_printoptions(precision=4)
CUDA = torch.cuda.is_available()
print("CUDA: ",CUDA)

test_lock = Lock()
model = None
testing_dataloader = None
cutoff_index_list, param_shape_list = [], []

class TestService(Service):
    def __init__(self) -> None:
        super().__init__()

        self.logging("Test Process Start Successfully")

    def exposed_test(self, tensor_list_received, round_id):
        tensor_list_received = [rpyc.classic.obtain(i).cuda() for i in tensor_list_received] if CUDA \
            else [rpyc.classic.obtain(i) for i in tensor_list_received]

        # t1 = time.time()
        testing_thread = Thread(target=test_thread, args=(round_id, tensor_list_received,))
        testing_thread.start()
        # testing_thread.join()
        # t2 = time.time()
        # self.logging("start_testing_thread: %.4f" % (t2-t1))
        

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Test Process] '+str(string))
        sys.stdout.flush()

def test_thread(round_id, tensor_list_received):
        global model, testing_dataloader, cutoff_index_list, param_shape_list, test_lock

        test_lock.acquire()
        accuracy = 0
        positive_test_number = 0.0
        total_test_number = 0.0
        # loss = 0.0
        ''' restore param '''
        for i, param in enumerate(model.parameters()):
            param.data = tensor_list_received[i]

        t1 = time.time()
        with torch.no_grad():
            for step, (test_x, test_y) in enumerate(testing_dataloader):
                if CUDA:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                test_output = model(test_x)
                # loss += self.loss_func(test_output, test_y)
                # pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                # positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
                pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
                total_test_number += len(test_y)
        # loss /= step
        # print("loss: %.4f" % loss)
        accuracy = positive_test_number / total_test_number
        t2 = time.time()
        logging('test_time: %.4f' % (t2-t1))
        logging(' - test - round_id: %d; accuracy: %.4f;' % (round_id, accuracy))
        test_lock.release()

def logging(string):
    print('['+str(datetime.datetime.now())+'] [Test Process] '+str(string))
    sys.stdout.flush()

if __name__ == "__main__":
    ''' Parse arguments and create profile. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    args = parser.parse_args()

    """ Local Training Profile """
    model_name, dataset_name = args.model, args.dataset
    local_profile = {
        'model_profile' : {
            'model_name' : model_name
        },
        'dataset_profile' : {
            'dataset_name' : dataset_name,
            'is_iid' : False,
            'world_size' : 100,
            'partition_rank' : 0,
            'batch_size' : 1024,
            'dirichlet_alpha' : 1
        }
    } 

    ''' Model Initialization'''
    model_manager = Model_Manager(local_profile['model_profile'])
    model = model_manager.load_model()
    # optimizer = model_manager.get_optimizer()
    # loss_func = model_manager.get_loss_func()

    ''' Initialization Index List '''
    param_shape_list = []
    cutoff_index_list = []
    s_index = 0
    for param in model.parameters():
        param_shape_list.append(param.data.shape)
        flatten_params = param.data.view(-1)
        e_index = s_index + len(flatten_params)
        cutoff_index_list.append([s_index, e_index])
        s_index = e_index


    ''' Dataset Initialization '''
    dataset_manager = Dataset_Manager(local_profile['dataset_profile'])
    testing_dataloader = dataset_manager.get_testing_dataloader()

    service = TestService()
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    server = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    server.start()