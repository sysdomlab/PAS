import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time, random

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from threading import Lock,Thread

path=os.path.dirname(__file__)
sys.path.append(path+'/../..')
from model_manager import Model_Manager
from dataset_manager import Dataset_Manager



torch.set_num_threads(4)
torch.set_printoptions(precision=4)
CUDA = torch.cuda.is_available()
print("CUDA: ",CUDA)

test_lock = Lock()
model_name = None
model_manager = None
model = None
loss_func = None
testing_dataloader = None
cutoff_index_list, param_shape_list = [], []

class TestService(Service):
    def __init__(self) -> None:
        super().__init__()

        self.logging("Test Process Start Successfully")

    def exposed_test(self, tensor_to_restore, round_id):
        tensor_to_restore = rpyc.classic.obtain(tensor_to_restore)

        # t1 = time.time()
        testing_thread = Thread(target=test_thread, args=(round_id, tensor_to_restore,))
        testing_thread.start()
        # testing_thread.join()
        # t2 = time.time()
        # self.logging("start_testing_thread: %.4f" % (t2-t1))
        

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Test Process] '+str(string))
        sys.stdout.flush()

def test_thread(round_id, tensor_to_restore):
        global model_manager, testing_dataloader, cutoff_index_list, param_shape_list, test_lock

        test_lock.acquire()
        if CUDA: tensor_to_restore = tensor_to_restore.cuda()
        ''' restore param '''
        model_manager.restore_model_with_tensor(tensor_to_restore)

        t1 = time.time()
        if model_name.startswith("Bert"):
            accuracy, loss = bert_test()
        else:
            accuracy, loss = test()
        t2 = time.time()
        logging('test_time: %.4f' % (t2-t1))
        logging(' - test - round_id: %d; accuracy: %.4f; loss: %.4f' % (round_id, accuracy, loss))
        test_lock.release()

def test():
    global model, testing_dataloader, loss_func
    positive_test_number = 0.0
    total_test_number = 0.0
    loss = 0.0
    with torch.no_grad():
        for step, (test_x, test_y) in enumerate(testing_dataloader):
            if CUDA:
                test_x = test_x.cuda()
                test_y = test_y.cuda()
            test_output = model(test_x)
            if model_name.startswith("Vit"):
                test_output = test_output.logits
            loss += loss_func(test_output, test_y).item()
            _, pred_y = torch.max(test_output, 1)
            positive_test_number += (pred_y == test_y).sum().item()
            total_test_number += len(test_y)
    return positive_test_number / total_test_number, loss

def bert_test():
    global model, testing_dataloader
    total_acc_val = 0
    total_loss_val = 0
    with torch.no_grad():
        for inputs, labels in testing_dataloader:
            input_ids = inputs['input_ids'].squeeze(1)
            masks = inputs['attention_mask']
            if CUDA:
                input_ids = input_ids.cuda()
                masks = masks.cuda()
                labels = labels.cuda()
            outputs = model(input_ids, attention_mask=masks, labels=labels)

            batch_loss = outputs.loss
            acc = (outputs.logits.argmax(dim=1) == labels).sum().item()
            total_acc_val += acc
            total_loss_val += batch_loss.item()
    avg_acc = total_acc_val / len(testing_dataloader.dataset)
    avg_loss = total_loss_val / len(testing_dataloader.dataset)
    return avg_acc, avg_loss


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
            'batch_size' : 256,
            'dirichlet_alpha' : 1
        }
    } 

    ''' Model Initialization'''
    model_manager = Model_Manager(local_profile['model_profile'])
    model = model_manager.load_model()
    # optimizer = model_manager.get_optimizer()
    loss_func = model_manager.get_loss_func()

    ''' Initialization Index List '''
    model_manager.get_param_shape()


    ''' Dataset Initialization '''
    dataset_manager = Dataset_Manager(local_profile['dataset_profile'])
    testing_dataloader = dataset_manager.get_testing_dataloader()

    service = TestService()
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    server = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    server.start()