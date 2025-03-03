import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random, datetime, sys, pprint, time
# from transformers import BertTokenizer
# from datasets import load_dataset


np.set_printoptions(4)

DATASET_PATH = "./"


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
    

class Bert20NewsDataset(Dataset):
    def __init__(self, dataset, tokenizer) -> None:
        super().__init__()
        self.datas = dataset['text']
        self.tokenizer = tokenizer
        self.targets = [label for label in dataset['label']]
    
    def __getitem__(self, index):
        inputs = self.tokenizer(self.datas[index], truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        return inputs, self.targets[index]

    def __len__(self):
        return len(self.datas)
    

class BertSstDataset(Dataset):
    def __init__(self, dataset, tokenizer) -> None:
        super().__init__()
        self.datas = dataset['sentence']
        self.tokenizer = tokenizer
        self.targets = [label for label in dataset['label']]
    
    def __getitem__(self, index):
        inputs = self.tokenizer(self.datas[index], truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        return inputs, self.targets[index]

    def __len__(self):
        return len(self.datas)
    

class BertImdbDataset(Dataset):
    def __init__(self, dataset, tokenizer) -> None:
        super().__init__()
        self.datas = dataset['text']
        self.tokenizer = tokenizer
        self.targets = [label for label in dataset['label']]
    
    def __getitem__(self, index):
        inputs = self.tokenizer(self.datas[index], truncation=True, padding='max_length', max_length=512, return_tensors='pt')

        return inputs, self.targets[index]

    def __len__(self):
        return len(self.datas)



class Dataset_Manager():
    def __init__(self, dataset_profile): # (dataset_name, is_iid, total_partition_number, partition_rank)
            self.dataset_name = dataset_profile['dataset_name']
            self.is_iid = dataset_profile['is_iid']
            self.world_size = dataset_profile['world_size']
            self.partition_rank = dataset_profile['partition_rank']
            self.dirichlet_alpha = dataset_profile['dirichlet_alpha']

            self.transform_to_224=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            self.batch_size = dataset_profile['batch_size']
            self.training_dataset, self.testing_dataset, self.class_num = self.get_dataset()
            self.local_size = len(self.training_dataset) / self.world_size
            self.logging(self.local_size)
            self.logging('create dataset')

    def logging(self, string, hyperparamiters=None):
        print('['+str(datetime.datetime.now())+'] [Dataset Manager] ' + str(string))
        if hyperparamiters != None:
            pprint.pprint(hyperparamiters)
        sys.stdout.flush()


    def get_dataset(self):
        if self.dataset_name == 'Mnist':
            train_dataset = datasets.MNIST(root=f'{DATASET_PATH}/Mnist', train=True, download=True, transform=transforms.ToTensor())
            test_dataset = datasets.MNIST(root=f'{DATASET_PATH}/Mnist', train=False, transform=transforms.ToTensor())
            class_num = 10
        if self.dataset_name == 'EMnist':
            train_dataset = datasets.EMNIST(root=f'{DATASET_PATH}/EMnist', split='balanced',train=True, download=True, transform=transforms.ToTensor())
            test_dataset = datasets.EMNIST(root=f'{DATASET_PATH}/EMnist', split="balanced",train=False, transform=transforms.ToTensor())
            class_num = 47
        if self.dataset_name == 'FashionMnist':
            train_dataset = datasets.FashionMNIST(root=f'{DATASET_PATH}/FashionMnist', train=True, download=True, transform=transforms.ToTensor())
            test_dataset = datasets.FashionMNIST(root=f'{DATASET_PATH}/FashionMnist', train=False, transform=transforms.ToTensor())
            class_num = 10
        if self.dataset_name == 'Cifar10':
            train_dataset = datasets.CIFAR10(root=f'{DATASET_PATH}/Cifar10', train=True, download=True, transform=transforms.ToTensor())
            test_dataset = datasets.CIFAR10(root=f'{DATASET_PATH}/Cifar10', train=False, transform=transforms.ToTensor())
            class_num = 10
        if self.dataset_name == 'KWS':
            train_dataset = KWSconstructor(root=f'./kws/index_train.txt', transform=None)
            test_dataset = KWSconstructor(root=f'./kws/index_test.txt', transform=None)
            class_num = 10
        if self.dataset_name == 'Cifar10_224':
            train_dataset = datasets.CIFAR10(root=f'{DATASET_PATH}/Cifar10', train=True, download=True, transform=self.transform_to_224)
            test_dataset = datasets.CIFAR10(root=f'{DATASET_PATH}/Cifar10', train=False, transform=self.transform_to_224)
            class_num = 10
            self.batch_size = 16
        if self.dataset_name == '20News':
            dataset_path = "/home/zgan/Datasets/PsyMatrix/cls_20newsgroups_SubjectTextVsLabel__BaseDefault"
            model_path = "/home/zgan/Models/google/bert_uncased_L-2_H-128_A-2"
            tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_path)
            train_dataset = Bert20NewsDataset(load_dataset(dataset_path, split="train"), tokenizer)
            test_dataset = Bert20NewsDataset(load_dataset(dataset_path, split="test"), tokenizer)
            self.batch_size = 32
            class_num = 20
        if self.dataset_name == 'sst2':
            dataset_path = "/home/zgan/Datasets/stanfordnlp/sst2"
            model_path = "/home/zgan/Models/google/bert_uncased_L-2_H-128_A-2"
            tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_path)
            train_dataset = BertSstDataset(load_dataset(dataset_path, split="train"), tokenizer)
            test_dataset = BertSstDataset(load_dataset(dataset_path, split="validation"), tokenizer)
            self.batch_size = 16
            class_num = 2
        if self.dataset_name == 'imdb':
            dataset_path = "/home/zgan/Datasets/stanfordnlp/imdb"
            model_path = "/home/zgan/Models/google/bert_uncased_L-2_H-128_A-2"
            tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_path)
            dataset = load_dataset(dataset_path, split="train")
            dataset = dataset.shuffle(0)
            dataset_size = len(dataset_size)
            train_size = int(dataset_size * 0.8)
            train_dataset = BertImdbDataset(dataset[:train_size], tokenizer)
            test_dataset = BertImdbDataset(dataset[train_size:], tokenizer)
            self.batch_size = 16
            class_num = 2
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
        composition_ratio = np.random.dirichlet(tuple(self.dirichlet_alpha/self.class_num for _ in range(self.class_num)), self.world_size)
        # if self.partition_rank == 0:
        #     self.logging(np.argmax(np.random.dirichlet(tuple(self.dirichlet_alpha for _ in range(self.class_num)), self.world_size),axis=1))
        return (composition_ratio*self.local_size).astype(int)

    # def get_index_partition(self): 
    #     ''' dirichelet random sampling '''
    #     local_idxs = []
    #     self.set_seed(0)
    #     labels = np.array(self.training_dataset.targets)
    #     sorted_idxs = np.argsort(labels)
    #     composition = self.get_dirichlet()
    #     composition = composition[self.partition_rank]
    #     self.logging(composition)
    #     # print('local dataset composition: ' + str(composition))
    #     class_pool_size = int(len(self.training_dataset) / self.class_num)
    #     self.set_seed(self.partition_rank)
    #     for i in range(len(composition)):
    #         for j in range(composition[i]):
    #             sample_index = sorted_idxs[int(class_pool_size*random.random()) + class_pool_size*i] # randomly sampling
    #             # sample_index = sorted_idxs[(class_pool_size/self.world_size*self.rank+j) % class_pool_size + class_pool_size*i]
    #             local_idxs.append(sample_index)
    #     self.set_seed(0)
    #     return local_idxs

    def get_index_partition(self): 
        ''' dirichelet without overlap '''
        local_idxs = np.array([]).astype(int)
        self.set_seed(0)
        labels = np.array(self.training_dataset.targets)
        sorted_idxs = np.argsort(labels)
        composition = self.get_dirichlet()
        class_accumulate = np.sum(composition[:self.partition_rank], axis=0)
        composition = composition[self.partition_rank]
        self.logging(composition)
        # print('local dataset composition: ' + str(composition))
        class_pool_size = int(len(self.training_dataset) / self.class_num)
        for i in range(self.class_num):
            # if class_accumulate[i]%class_pool_size <= (class_accumulate[i]+composition[i]) % class_pool_size:
            #     client_class_index = sorted_idxs[class_pool_size*i+(class_accumulate[i])%class_pool_size:class_pool_size*i+(class_accumulate[i]+composition[i])%class_pool_size]
            # else:
            #     client_class_index = np.concatenate((sorted_idxs[class_pool_size*i+class_accumulate[i]%class_pool_size:class_pool_size*(i+1)], \
            #                                          sorted_idxs[class_pool_size*i:class_pool_size*i+(class_accumulate[i]+composition[i])%class_pool_size]),axis=0)
            client_class_index = np.array([]).astype(int)
            class_accumulate[i] = class_accumulate[i]%class_pool_size
            while class_accumulate[i] + composition[i] > class_pool_size:
                client_class_index = np.concatenate((client_class_index, sorted_idxs[class_pool_size*i+class_accumulate[i]:class_pool_size*(i+1)]),axis=0)
                composition[i] -= (class_pool_size-class_accumulate[i])
                class_accumulate[i] = 0
            client_class_index = np.concatenate((client_class_index, sorted_idxs[class_pool_size*i+class_accumulate[i]:class_pool_size*i+class_accumulate[i]+composition[i]]),axis=0)
            # self.logging(len(client_class_index))
            local_idxs = np.concatenate((local_idxs, client_class_index), axis = 0)
            local_idxs = local_idxs.astype(int)
        return local_idxs

    # def get_index_partition(self):
    #     labels = np.array(self.training_dataset.targets) #pytorch老版本train_labels，新版本targets
    #     idxs = np.argsort(labels)
    #     if self.world_size >= 10 and self.class_num == 10:
    #         l = [i for i in range(2*self.world_size)]
    #         single_class_parts = 2*self.world_size/self.class_num
    #         random.shuffle(l)
    #         if self.partition_rank == 0:
    #             a = 0
    #             for i in range(self.world_size):
    #                 if l[2*i]/single_class_parts == l[2*i+1]/single_class_parts: a += 1
    #             self.logging('bad_partition: %d' % a)
    #         each_part_size = len(self.training_dataset) // (2 * self.world_size)
    #         final_idxs = np.concatenate((idxs[each_part_size*l[2*self.partition_rank] : each_part_size*(l[2*self.partition_rank]+1)], \
    #                                      idxs[each_part_size*l[2*self.partition_rank+1] : each_part_size*(l[2*self.partition_rank+1]+1)]), axis=0)
    #         return final_idxs
    #     else:
    #         each_part_size = len(self.training_dataset) // self.world_size
    #         return idxs[each_part_size*self.partition_rank : each_part_size*(self.partition_rank+1)]

    def get_iid_index_partition(self):
        labels = np.array(self.training_dataset.targets) #pytorch老版本train_labels，新版本targets
        idxs = np.argsort(labels)
        np.random.shuffle(idxs)
        each_part_size = len(self.training_dataset) // self.world_size
        return idxs[each_part_size*self.partition_rank : each_part_size*(self.partition_rank+1)]


    def get_training_dataloader(self):
        if self.is_iid:
            self.set_seed(0)
            index_partition = self.get_iid_index_partition()
            training_dataloader = DataLoader(DatasetSplit(self.training_dataset, index_partition), 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        else:
            self.set_seed(0)
            index_partition = self.get_index_partition()
            training_dataloader = DataLoader(DatasetSplit(self.training_dataset, index_partition),
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        return training_dataloader

    def get_testing_dataloader(self):
        self.set_seed(0)
        testing_dataloader = DataLoader(dataset=self.testing_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True)
        return testing_dataloader


if __name__ == "__main__":
    pass