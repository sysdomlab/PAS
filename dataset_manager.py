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
            self.world_size = dataset_profile['world_size']
            self.partition_rank = dataset_profile['partition_rank']
            self.dirichlet_alpha = dataset_profile['dirichlet_alpha']

            self.transform_to_224=transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                    ])

            self.batch_size = dataset_profile['batch_size']
            self.training_dataset, self.testing_dataset, self.class_num = self.get_dataset()
            self.local_size = len(self.training_dataset) / self.world_size
            self.logging(self.local_size/5)
            self.logging('create dataset')

    def logging(self, string, hyperparamiters=None):
        print('['+str(datetime.datetime.now())+'] [Dataset Manager] ' + str(string))
        if hyperparamiters != None:
            pprint.pprint(hyperparamiters)
        sys.stdout.flush()


    def get_dataset(self):
        if self.dataset_name == 'Mnist':
            train_dataset = datasets.MNIST(root='./Datasets/Mnist', train=True, download=True, transform=transforms.ToTensor())
            test_dataset = datasets.MNIST(root='./Datasets/Mnist', train=False, transform=transforms.ToTensor())
            class_num = 10
        if self.dataset_name == 'EMnist':
            train_dataset = datasets.EMNIST(root='./Datasets/EMnist', split='balanced',train=True, download=True, transform=transforms.ToTensor())
            test_dataset = datasets.EMNIST(root='./Datasets/EMnist', split="balanced",train=False, transform=transforms.ToTensor())
            class_num = 47
        if self.dataset_name == 'FashionMnist':
            train_dataset = datasets.FashionMNIST(root='./Datasets/FashionMnist', train=True, download=True, transform=transforms.ToTensor())
            test_dataset = datasets.FashionMNIST(root='./Datasets/FashionMnist', train=False, transform=transforms.ToTensor())
            class_num = 10
        if self.dataset_name == 'Cifar10':
            train_dataset = datasets.CIFAR10(root='./Datasets/Cifar10', train=True, download=True, transform=transforms.ToTensor())
            test_dataset = datasets.CIFAR10(root='./Datasets/Cifar10', train=False, transform=transforms.ToTensor())
            class_num = 10
        if self.dataset_name == 'KWS':
            train_dataset = KWSconstructor(root='./kws/index_train.txt', transform=None)
            test_dataset = KWSconstructor(root='./kws/index_test.txt', transform=None)
            class_num = 10
        if self.dataset_name == 'Cifar10_224':
            train_dataset = datasets.CIFAR10(root='./Datasets/Cifar10_224', train=True, download=True, transform=self.transform_to_224)
            test_dataset = datasets.CIFAR10(root='./Datasets/Cifar10_224', train=False, transform=self.transform_to_224)
            class_num = 10
        return train_dataset, test_dataset, class_num

    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False

    def get_dirichlet(self):
        composition_ratio = np.random.dirichlet(tuple(self.dirichlet_alpha for _ in range(10)), self.world_size)[self.partition_rank]
        # if self.partition_rank == 0:
        #     self.logging(np.argmax(np.random.dirichlet(tuple(self.dirichlet_alpha for _ in range(self.class_num)), self.world_size),axis=1))
        return (composition_ratio*self.local_size).astype(int)

    def get_index_partition(self):
        local_idxs = []
        class_list = []
        self.set_seed(0)
        for i in range(self.world_size):
            temp_list = []
            for i in range(10):
                temp_list.append(random.randint(1,self.class_num))
            class_list.append(temp_list)
        self.logging(class_list)
        class_list = class_list[self.partition_rank]
        labels = np.array(self.training_dataset.targets)
        sorted_idxs = np.argsort(labels)
        composition = self.get_dirichlet()
        # print('local dataset composition: ' + str(composition))
        class_pool_size = int(len(self.training_dataset) / self.class_num)
        for i in range(len(class_list)):
            for j in range(composition[i]):
                sample_index = sorted_idxs[int(class_pool_size*random.random()) + class_pool_size*(class_list[i]-1)] # randomly sampling
                # sample_index = sorted_idxs[(class_pool_size/self.world_size*self.rank+j) % class_pool_size + class_pool_size*i]
                local_idxs.append(sample_index)
        return local_idxs

    # def get_index_partition(self):
    #     labels = np.array(self.training_dataset.targets) #pytorch老版本train_labels，新版本targets
    #     idxs = np.argsort(labels)
    #     each_part_size = len(self.training_dataset) // self.world_size
    #     return idxs[each_part_size*self.partition_rank : each_part_size*(self.partition_rank+1)]


    def get_training_dataloader(self):
        if self.is_iid:
            self.set_seed(self.partition_rank)
            training_dataloader = DataLoader(self.training_dataset, 
                                            batch_size=self.batch_size, 
                                            shuffle=True, 
                                            num_workers=self.world_size, 
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