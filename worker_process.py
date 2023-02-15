import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time

from model_manager import Model_Manager
from dataset_manager import Dataset_Manager
from sync_manager import Sync_Manager
from MultiProcess_CAS import ParamGIFT_Manager
from LinearCAS_manager import LinearCAS_Manager
# from free_ParamGIFT_manager import FreeParamGIFT_Manager
# from new_ParamGIFT_manager import ParamGIFT_Manager

CUDA = torch.cuda.is_available()
print("CUDA: ",CUDA)

class Client():
    def __init__(self, local_profile, sync_prefile):
        ''' Variables required for local training. '''
        self.model_manager = Model_Manager(local_profile['model_profile'])
        self.model = self.model_manager.load_model()
        self.optimizer = self.model_manager.get_optimizer()
        self.loss_func = self.model_manager.get_loss_func()
        # self.model_manager.load_exist_model(170)

        self.dataset_manager = Dataset_Manager(local_profile['dataset_profile'])
        self.training_dataloader = self.dataset_manager.get_training_dataloader()
        self.testing_dataloader = self.dataset_manager.get_testing_dataloader()
        self.sync_frequency = sync_prefile['sync_frequency']
        self.round_id = 0
        self.iter_id = 0
        self.epoch_id = 0
        self.max_epoch = 3000

        """ Sync-related variables """
        if sync_profile['interlayer_type'] == 'ParamGIFT':
            self.sync_manager = ParamGIFT_Manager(self.model, sync_prefile)
        # elif sync_profile['interlayer_type'] == 'freeParamGIFT':
        #     self.sync_manager = FreeParamGIFT_Manager(self.model, sync_prefile)
        elif sync_profile['interlayer_type'] == 'LinearCAS':
            self.sync_manager = LinearCAS_Manager(self.model, sync_prefile)
        else:
            self.sync_manager = Sync_Manager(self.model, sync_prefile)
        self.rank = sync_profile['dist_profile']['rank']

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
                pred_y = torch.max(test_output, 1)[1]
                positive_test_number += (pred_y == test_y).sum().item()
                total_test_number += len(test_y)
        # loss /= step
        # print("loss: %.4f" % loss)
        accuracy = positive_test_number / total_test_number
        return accuracy


    def train(self):
        print('\n\n\t-------- START TRAINING --------\n')
        while self.epoch_id < self.max_epoch:
            # print(next(self.model.parameters()).device)
            self.epoch_id += 1
            self.logging('start epoch: %d' % self.epoch_id)
            for step, (b_x, b_y) in enumerate(self.training_dataloader):
                self.iter_id += 1
                if CUDA:
                    b_x = b_x.cuda()
                    b_y = b_y.cuda()
                self.optimizer.zero_grad()
                self.loss_func(self.model(b_x)+1e-8, b_y).backward()
                self.optimizer.step()
                if self.sync_manager.try_sync_model(self.iter_id):
                    if self.iter_id % (max(64, self.sync_frequency)) == 0 and self.rank == 0:
                        accuracy = self.test()
                        self.logging(' - test - iter_id: %d; epoch_id: %d, round_id: %d; accuracy: %.4f;' % (self.iter_id, self.epoch_id, self.sync_manager.sync_round_id, accuracy))
                    # numpy.save('./NPY/param_client_%d_epoch_%d' % (self.rank, epoch_id), list(self.model.parameters())[0][0][0].detach().cpu().numpy())
                        self.model_manager.save_model(self.sync_manager.sync_round_id)
            self.logging('finish epoch: %d\n' % self.epoch_id)


if __name__ == '__main__':
    
    """ Parse arguments and create GIFT profile. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_address', type=str, default='127.0.0.1')
    parser.add_argument('--world_size', type=int, default=5)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')
    parser.add_argument('--model', type=str, default='Model Missing...')
    parser.add_argument('--dataset', type=str, default='Dataset Missing...')
    parser.add_argument('--interlayer_type', type=str, default='Interlayer Missing...')
    parser.add_argument('--batchsize', type=int, default=20)
    parser.add_argument('--init_frequency', type=int, default=128)
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5)

    args = parser.parse_args()
    if CUDA:
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
    print('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks + '\n')

    """ Local Training Profile """
    # model_name, dataset_name, is_iid = 'CNN_Cifar10', 'Cifar10', False
    # model_name, dataset_name, is_iid = 'CNN_Mnist', 'Mnist', False
    # model_name, dataset_name, is_iid = 'CNN_FashionMnist', 'FashionMnist', False
    # model_name, dataset_name, is_iid = 'CNN_EMnist', 'EMnist', False
    # model_name, dataset_name, is_iid = 'VGG16_Cifar10', 'Cifar10', False
    # model_name, dataset_name, is_iid = 'ResNet18_Cifar10', 'Cifar10', False
    # model_name, dataset_name, is_iid = 'LSTM_KWS', 'KWS', False
    # model_name, dataset_name, is_iid = 'MobileNetV2_Cifar10', 'Cifar10', False
    # model_name, dataset_name, is_iid = 'GhostNet_Cifar10', 'Cifar10', False
    # model_name, dataset_name, is_iid = 'Ghost_ResNet56_Cifar10', 'Cifar10', False
    model_name, dataset_name, is_iid = args.model, args.dataset, False

    local_training_profile = {
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
    pprint.pprint(local_training_profile)

    """ Synchronization Profile """
    sync_frequency = args.init_frequency
    # interlayer_type = 'ParamGIFT' # GIFT, ParamGIFT, Default, LayerGIFT
    # interlayer_type = 'freeParamGIFT' # GIFT, ParamGIFT, Default, LayerGIFT
    # interlayer_type = 'LayerGIFT' # GIFT, ParamGIFT, Default, LayerGIFT
    # interlayer_type = 'GIFT' # GIFT, ParamGIFT, Default, LayerGIFT
    # interlayer_type = 'Default' # GIFT, ParamGIFT, Default, LayerGIFT
    interlayer_type = args.interlayer_type # GIFT, ParamGIFT, Default, LayerGIFT
    sync_profile = {
        'sync_frequency' : sync_frequency,
        'interlayer_type' : interlayer_type,
        'dist_profile' : {
            'master_address' : args.master_address,
            'world_size' : args.world_size,
            'rank' : args.rank
        }
    }
    print('- Sync Profile: ')
    pprint.pprint(sync_profile)

    """ Launch Training """
    client = Client(local_training_profile, sync_profile)
    client.train()


