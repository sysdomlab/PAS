import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random, datetime, sys, pprint

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]

class KWSconstructor(Dataset):
    def __init__(self, root, transform=None) -> None:
        f = open(root, 'r')
        data = []
        self.targets = []
        for line in f:
            s = line.split('\n')
            info = s[0].split(' ')
            data.append( (info[0],int(info[1])) )
            self.targets.append(int(info[1]))
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
    def __init__(self, dataset_profile): # (dataset_name, is_iid, total_partition_number, partition_rank)
            self.dataset_name = dataset_profile['dataset_name']
            self.is_iid = dataset_profile['is_iid']
            self.total_partition_number = dataset_profile['total_partition_number']
            self.partition_rank = dataset_profile['partition_rank']

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
            dataset = datasets.EMNIST(root='./Datasets/EMnist', split='balanced',train=True, download=True, transform=transforms.ToTensor())
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
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_index_partition(self):
        labels = np.array(self.training_dataset.targets) #pytorch老版本train_labels，新版本targets
        idxs = np.argsort(labels)
        each_part_size = len(self.training_dataset) // self.total_partition_number
        return idxs[each_part_size*self.partition_rank : each_part_size*(self.partition_rank+1)]

    def get_training_dataloader(self):
        if self.is_iid:
            self.set_seed(self.partition_rank)
            training_dataloader = DataLoader(self.training_dataset, 
                                            batch_size=self.batch_size, 
                                            shuffle=True, 
                                            num_workers=self.total_partition_number, 
                                            pin_memory=True)
        else:
            self.set_seed(0)
            index_partition = self.get_index_partition()
            training_dataloader = DataLoader(DatasetSplit(self.training_dataset, index_partition),
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        return training_dataloader

    def get_testing_dataloader(self):
        testing_dataloader = DataLoader(dataset=self.testing_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True)
        return testing_dataloader