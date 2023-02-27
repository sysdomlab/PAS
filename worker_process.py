import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from model_manager import Model_Manager
from dataset_manager import Dataset_Manager
from sync_manager import Sync_Manager
from client_rpyc import CAS_Manager

torch.set_printoptions(precision=20)
CUDA = torch.cuda.is_available()
print("CUDA: ",CUDA)

class Client():
    def __init__(self, local_profile, sync_prefile):
        ''' Variables required for local training. '''
        self.rank = sync_profile['rank']
        self.init_period = sync_prefile['init_period']
        self.max_epoch = 3000

        ''' Model Initialization'''
        self.model_manager = Model_Manager(local_profile['model_profile'])
        self.model = self.model_manager.load_model()
        self.optimizer = self.model_manager.get_optimizer()
        self.loss_func = self.model_manager.get_loss_func()

        ''' Dataset Initialization '''
        self.dataset_manager = Dataset_Manager(local_profile['dataset_profile'])
        self.training_dataloader = self.dataset_manager.get_training_dataloader()
        self.testing_dataloader = self.dataset_manager.get_testing_dataloader()


        """ Sync-related variables """
        if sync_profile['interlayer_type'] == 'CAS':
            self.sync_manager = CAS_Manager(self.model, sync_prefile)
        else:
            self.sync_manager = Sync_Manager(self.model, sync_prefile)

    def logging(self, string):
        print('[' + str(datetime.datetime.now()) + '] [Client] ' + str(string))
        sys.stdout.flush()

    def test(self):
        accuracy = 0
        positive_test_number = 0.0
        total_test_number = 0.0
        loss = 0.0
        with torch.no_grad():
            for step, (test_x, test_y) in enumerate(self.testing_dataloader):
                if CUDA:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                test_output = self.model(test_x)
                # loss += self.loss_func(test_output, test_y)
                # pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                # positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
                pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
                total_test_number += len(test_y)
        # loss /= step
        # print("loss: %.4f" % loss)
        accuracy = positive_test_number / total_test_number
        return accuracy


    def train(self):
        print('\n\n\t-------- START TRAINING --------\n')
        iter_id, round_id, epoch_id = 0, 0, 0
        while epoch_id < self.max_epoch:
            # print(next(self.model.parameters()).device)
            epoch_id += 1
            self.logging('start epoch: %d' % epoch_id)
            for step, (b_x, b_y) in enumerate(self.training_dataloader):
                self.logging(b_x)
                self.logging(b_y)
                iter_id += 1
                if CUDA:
                    b_x = b_x.cuda()
                    b_y = b_y.cuda()
                self.optimizer.zero_grad()
                # t1 = time.time()
                out_put = self.model(b_x)
                self.logging(out_put)
                loss = self.loss_func(out_put,b_y)
                self.logging(loss)
                loss.backward()
                # self.loss_func(self.model(b_x), b_y).backward()
                self.optimizer.step()
                tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
                for p in self.model.parameters():
                    flattened_param = p.data.view(-1)
                    tensor_of_flattened_params = torch.cat((tensor_of_flattened_params, flattened_param),0)
                self.logging(('iter_id: %d, step:' % iter_id)+str(step))
                self.logging(tensor_of_flattened_params)
                t2 = time.time()
                # self.logging('training_time: %.4f' % (t2-t1))
                # self.logging('finish local iteration: %d' % iter_id)
                if self.sync_manager.try_sync_model(iter_id):
                    t3 = time.time()
                    self.logging('try_sync_time: %.4f' % (t3-t2))
                    round_id += 1
                    if iter_id % max(self.init_period, 64) == 0 and self.rank == 0:
                        t1 = time.time()
                        accuracy = self.test()
                        t2 = time.time()
                        self.logging('test_time: %.4f' % (t2-t1))
                        self.logging(' - test - iter_id: %d; epoch_id: %d, round_id: %d; accuracy: %.4f;' % (iter_id, epoch_id, self.sync_manager.sync_round_id, accuracy))
                        # self.model_manager.save_model(self.sync_manager.sync_round_id)
            self.logging('finish epoch: %d\n' % epoch_id)


if __name__ == '__main__':
    
    """ Parse arguments and create GIFT profile. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=str, default=20000)
    parser.add_argument('--sync_process_port', type=str, default=30000)
    parser.add_argument('--world_size', type=int, default=5)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')
    parser.add_argument('--model', type=str, default='Model Missing...')
    parser.add_argument('--dataset', type=str, default='Dataset Missing...')
    parser.add_argument('--interlayer_type', type=str, default='Interlayer Missing...')
    parser.add_argument('--batchsize', type=int, default=20)
    parser.add_argument('--init_period', type=int, default=128)
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5)

    args = parser.parse_args()
    if CUDA:
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
    print('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks + '\n')

    """ Local Training Profile """
    model_name, dataset_name, is_iid = args.model, args.dataset, False

    local_profile = {
        'model_profile' : {
            'model_name' : model_name
        },
        'dataset_profile' : {
            'dataset_name' : dataset_name,
            'is_iid' : is_iid,
            'world_size' : 1 if is_iid else args.world_size,
            'partition_rank' : 0 if is_iid else args.rank,
            'batch_size' : args.batchsize,
            'dirichlet_alpha' : args.dirichlet_alpha
        }
    }
    print('- Local Training Profile: ')
    pprint.pprint(local_profile)

    """ Synchronization Profile """
    interlayer_type = args.interlayer_type # GIFT, ParamGIFT, Default, LayerGIFT
    sync_profile = {
        'rank' : args.rank,
        'init_period' : args.init_period,
        'interlayer_type' : interlayer_type,
        'server_ip' : args.server_ip,
        'server_port' : args.server_port,
        'sync_process_port' : args.sync_process_port
        
    }
    print('- Sync Profile: ')
    pprint.pprint(sync_profile)

    """ Launch Training """
    client = Client(local_profile, sync_profile)
    client.train()


