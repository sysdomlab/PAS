import torch
import datetime, sys, numpy, pprint
from model_manager import Model_Manager
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np

device = torch.device("cuda")

class KWSconstructor(Dataset):
    def __init__(self, root, transform=None) -> None:
        f = open(root, 'r')
        data = []
        self.train_labels = []
        for line in f:
            s = line.split('\n')
            info = s[0].split(' ')
            data.append( (info[0],int(info[1])) )
            self.train_labels.append(int(info[1]))
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        f, label = self.data[index]
        feature = np.loadtxt(f)
        feature = np.reshape(feature, (50,10))
        feature = feature.astype(np.float32)
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label

    def __len__(self):
        return len(self.data)

class Dataset_Manager():
    def __init__(self, dataset_profile):
            self.dataset_name = dataset_profile['dataset_name']

            self.transform_to_224=transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                    ])

            self.batch_size = dataset_profile['batch_size']
            self.training_dataset = self.get_training_dataset()
            self.testing_dataset = self.get_testing_dataset()

            self.logging('create dataset')

    def logging(self, string, hyperparamiters=None):
        print('['+str(datetime.datetime.now())+'] [Dataset Manager] ' + str(string))
        if hyperparamiters != None:
            pprint.pprint(hyperparamiters)
        sys.stdout.flush()


    def get_training_dataset(self):
        if self.dataset_name == 'Mnist':
            dataset = datasets.MNIST(root='./Datasets/Mnist', train=True, download=True, transform=transforms.ToTensor())
        if self.dataset_name == 'EMnist':
            dataset = datasets.EMNIST(root='./Datasets/EMnist', split="balanced",train=True, download=True, transform=transforms.ToTensor()) 
        if self.dataset_name == 'FashionMnist':
            dataset = datasets.FashionMNIST(root='./Datasets/FashionMnist', train=True, download=True, transform=transforms.ToTensor())
        if self.dataset_name == 'Cifar10':
            dataset = datasets.CIFAR10(root='./Datasets/Cifar10', train=True, download=True, transform=transforms.ToTensor())
        if self.dataset_name == 'KWS':
            dataset = KWSconstructor(root='./kws/index_train.txt', transform=None)
        if self.dataset_name == 'Cifar10_224':
            dataset = datasets.CIFAR10(root='./Datasets/Cifar10_224', train=True, download=True, transform=self.transform_to_224)
        return dataset
    
    def get_testing_dataset(self):
        if self.dataset_name == 'Mnist':
            dataset = datasets.MNIST(root='./Datasets/Mnist', train=False, transform=transforms.ToTensor())
        if self.dataset_name == 'EMnist':
            dataset = datasets.EMNIST(root='./Datasets/EMnist', split="balanced",train=False, transform=transforms.ToTensor()) 
        if self.dataset_name == 'FashionMnist':
            dataset = datasets.FashionMNIST(root='./Datasets/FashionMnist', train=False, transform=transforms.ToTensor())
        if self.dataset_name == 'Cifar10':
            dataset = datasets.CIFAR10(root='./Datasets/Cifar10', train=False, transform=transforms.ToTensor())
        if self.dataset_name == 'KWS':
            dataset = KWSconstructor(root='./kws/index_test.txt', transform=None)
        if self.dataset_name == 'Cifar10_224':
            dataset = datasets.CIFAR10(root='./Datasets/Cifar10_224', train=False, transform=self.transform_to_224)
        return dataset

    def get_training_dataloader(self):
        training_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)
        return training_dataloader

    def get_testing_dataloader(self):
        testing_dataloader = DataLoader(dataset=self.testing_dataset, batch_size=self.batch_size, shuffle=True)
        return testing_dataloader

def logging(string):
    print('[' + str(datetime.datetime.now()) + '] [Client] ' + str(string))
    sys.stdout.flush()

if __name__ == '__main__':

    """Training Profile"""
    # model_name, dataset_name = 'MobileNetV2_Cifar10', 'Cifar10'
    model_name, dataset_name = 'ResNet18_Cifar10', 'Cifar10'
    # model_name, dataset_name = 'Ghost_ResNet56_Cifar10', 'Cifar10'
    # model_name, dataset_name = 'GhostNet_Cifar10', 'Cifar10'
    # model_name, dataset_name = 'CNN_Cifar10', 'Cifar10'
    # model_name, dataset_name = 'CNN_EMnist', 'EMnist'
    # model_name, dataset_name = 'CNN_FashionMnist', 'FashionMnist'

    training_profile = {
        'model_profile' : {
            'model_name' : model_name
        },
        'dataset_profile' : {
            'dataset_name' : dataset_name,
            'batch_size' : 256

        }
    }
    print('- Training Profile: ')
    pprint.pprint(training_profile)

    """dataset manager"""
    dataset_manager = Dataset_Manager(training_profile['dataset_profile'])
    training_dataloader = dataset_manager.get_training_dataloader()
    testing_dataloader = dataset_manager.get_testing_dataloader()

    """model manager"""
    model_manager = Model_Manager(training_profile['model_profile'])
    model = model_manager.load_model().to(device)
    print(next(model.parameters()).device)
    optimizer = model_manager.get_optimizer()
    loss_func = model_manager.get_loss_func()

    max_epoch = 3000
    iter_id, epoch_id = 0, 0

    """training"""
    print('\n\n\t--------- START TRAINING ---------\n')
    while epoch_id < max_epoch:
        epoch_id += 1
        logging('start epoch: %d' % epoch_id)
        for step, (b_x, b_y) in enumerate(training_dataloader):
            iter_id += 1
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            optimizer.zero_grad()
            out_put = model(b_x)
            loss_func(out_put+1e-8, b_y).backward()
            optimizer.step()


        '''testing'''
        accuracy = 0
        positive_test_number = 0.0
        total_test_number = 0.0
        with torch.no_grad():
            for step, (test_x, test_y) in enumerate(testing_dataloader):
                test_x = test_x.to(device)
                test_y = test_y.to(device)
                test_output = model(test_x)
                pred_y = torch.max(test_output, 1)[1]
                positive_test_number += (pred_y == test_y).sum().item()
                total_test_number += len(test_y)
        accuracy = positive_test_number / total_test_number
        logging(' - test - iter_id: %d; epoch_id: %d; accuracy: %.4f;' % (iter_id, epoch_id, accuracy))
        logging('finishi epoch: %d\n' % epoch_id)