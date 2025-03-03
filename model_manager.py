import torch
from typing import Tuple, List
from models import CNN, LSTM, ResNet, VGG, AlexNet, DenseNet, LogisticRegression, MobileNetV2, GhostNet, Ghost_ResNet56
# from transformers import BertForSequenceClassification, ViTForImageClassification
import os, datetime, sys, pprint
import random
import numpy as np

CUDA = torch.cuda.is_available()

''' Suggested hyper-parameter values of (learning rate, weight decay, optimizer, loss_func) for each model. '''
ModelHyperparameterDict = { 
        'LogisticRegression_Mnist': 
            {'lr':0.01, 'wd':0, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'CNN_Mnist': 
            {'lr':0.01, 'wd':0.01, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'CNN_FashionMnist': 
            {'lr':0.01, 'wd':0.01, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'CNN_EMnist': 
            {'lr':0.01, 'wd':0.01, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'CNN_Cifar10':
            {'lr':0.01, 'wd':0.01, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'CNN_KWS':
            {'lr':0.01, 'wd':0.01, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'SVM_Points':
            {'lr':0.01, 'wd':0, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'LSTM_KWS':
            {'lr':0.01, 'wd':0.01, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'ResNet18_Cifar10':
            {'lr':0.1, 'wd':0.001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'ResNet20_Cifar10':
            {'lr':0.1, 'wd':0.001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'VGG16_Cifar10':
            {'lr':0.1, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'VGG11_Cifar10':
            {'lr':0.1, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'AlexNet_Cifar10':
            {'lr':0.1, 'wd':0.001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'AlexNet_ImageNet':
            {'lr':0.1, 'wd':0.001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'DenseNet121_ImageNet':
            {'lr':0.1, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'MobileNetV2_Cifar10':
            {'lr':0.1, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'GhostNet_Cifar10':
            {'lr':0.1, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'Ghost_ResNet56_Cifar10':
            {'lr':0.1, 'wd':0.0001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
        'Bert_20News':
            {'lr':0.01, 'wd': 0.01, 'optimizer': torch.optim.SGD, 'loss_func': None},
        'Bert_sst2':
            {'lr':1e-4, 'wd': 0.01, 'optimizer': torch.optim.Adam, 'loss_func': None},   
        'Bert_imdb':
            {'lr':1e-4, 'wd': 0.01, 'optimizer': torch.optim.Adam, 'loss_func': None},   
        'Vit_Cifar10_224':
            {'lr':0.1, 'wd': 0.001, 'optimizer': torch.optim.SGD, 'loss_func': torch.nn.CrossEntropyLoss()},
    }

class Model_Manager():
    def __init__(self, model_profile):
        self.model_name = model_profile['model_name']
        self.hyperparameters = ModelHyperparameterDict[model_profile['model_name']]

        self.model = None
        self.optimizer = None
        self.loss_func = None

        self.param_shape_list: List[torch.Size] = []
        self.cutoff_index_list: List[Tuple[int, int]] = []

        self.logging('create model manager', self.hyperparameters) # 

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Model Manager] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
            print
        sys.stdout.flush()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False

    def get_optimizer(self):
        self.optimizer = self.hyperparameters['optimizer'](self.model.parameters(), 
                                                        lr=self.hyperparameters['lr'], 
                                                        weight_decay=self.hyperparameters['wd'])
        return self.optimizer

    def get_loss_func(self):
        self.loss_func = self.hyperparameters['loss_func']
        return self.loss_func

    def load_model(self, CHECKPOINT_ENABLED=True):
        self.load_model_architecture()
        self.try_init_model_from_checkpoint(CHECKPOINT_ENABLED)
        self.model = self.model.cuda() if CUDA else self.model
        return self.model

    def load_model_architecture(self):
        """Load the model architecture"""
        if self.model_name == 'LogisticRegression_Mnist':
            self.model = LogisticRegression.LogisticRegression()
        if self.model_name == 'CNN_Mnist':
            self.model = CNN.CNN_Mnist()
        if self.model_name == 'CNN_FashionMnist':
            self.model = CNN.CNN_FashionMnist()
        if self.model_name == 'CNN_EMnist':
            self.model = CNN.CNN_EMnist()
        if self.model_name == 'CNN_Cifar10':
            self.model = CNN.CNN_Cifar10()
        if self.model_name == 'CNN_KWS':
            self.model = CNN.CNN_KWS()
        if self.model_name == 'LSTM_KWS':
            self.model = LSTM.LSTM_KWS()
        if self.model_name == 'ResNet18_Cifar10':
            self.model = ResNet.ResNet18_Cifar10()
        if self.model_name == 'ResNet20_Cifar10':
            self.model = ResNet.ResNet20_Cifar10()
        if self.model_name == 'VGG16_Cifar10':
            self.model = VGG.VGG16_Cifar10()
        if self.model_name == 'VGG11_Cifar10':
            self.model = VGG.VGG11_CIFAR10()
        if self.model_name == 'AlexNet_Cifar10':
            self.model = AlexNet.AlexNet_Cifar10() 
        if self.model_name == 'AlexNet_ImageNet':
            self.model = AlexNet.AlexNet_ImageNet() 
        if self.model_name == 'DenseNet121_ImageNet':
            self.model = DenseNet.DenseNet121_ImageNet()
        if self.model_name == 'MobileNetV2_Cifar10':
            self.model = MobileNetV2.MobileNetV2_Cifar10()
        if self.model_name == 'GhostNet_Cifar10':
            self.model = GhostNet.ghostnet()
        if self.model_name == 'Ghost_ResNet56_Cifar10':
            self.model = Ghost_ResNet56.Ghost_resnet56()

        self.set_seed(0)
        if self.model_name == 'Vit_Cifar10_224':
            self.model = ViTForImageClassification.from_pretrained('/home/zgan/Datasets/google/vit-base-patch16-224')
        
        if self.model_name == 'Bert_20News':
            # model_path = "/home/zgan/Models/google-bert/bert-base-uncased"
            # self.model = BertForSequenceClassification.from_pretrained(
            #     model_path,
            #     num_labels=20,
            #     output_attentions = False,
            #     output_hidden_states = False,
            #     )
            # from peft import LoraConfig, TaskType, get_peft_model
            # lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=4)
            # self.model = get_peft_model(self.model, lora_config)
            model_path = "/home/zgan/Models/google/bert_uncased_L-2_H-128_A-2"
            self.model = BertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=20,
                output_attentions = False,
                output_hidden_states = False,
                )

        if self.model_name in ['Bert_sst2', 'Bert_imdb']:
            model_path = "/home/zgan/Models/google/bert_uncased_L-2_H-128_A-2"
            self.model = BertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,
                output_attentions = False,
                output_hidden_states = False,
                )
        
    def try_init_model_from_checkpoint(self, CHECKPOINT_ENABLED):
        if self.model_name.startswith("Bert") or self.model_name.startswith("Vit"):
            return
        if CHECKPOINT_ENABLED:
            checkpoint_name = self.model_name + '_checkpoint_0.t7'
            if os.path.exists(checkpoint_name):
                self.logging('recover model from: ' + checkpoint_name)
                checkpoint = torch.load(checkpoint_name)
                self.model.load_state_dict(checkpoint['state'])
            else:
                self.save_model(0)
                self.logging('no checkpoint found; randomly initialize model and save it to: ' + checkpoint_name)

    def load_exist_model(self, round_id):
        checkpoint_name = self.model_name + '_checkpoint_' + str(round_id) + '.t7'
        if os.path.exists("./NPY/" + checkpoint_name):
            self.logging('recover model from: ' + checkpoint_name)
            checkpoint = torch.load("./NPY/" + checkpoint_name)
            self.model.load_state_dict(checkpoint['state'])
        else:
            print("no checkpoint found!")
            return ValueError

    def save_model(self, round_id):
        state = {'state': self.model.state_dict(), 'round': round_id}
        checkpoint_name = self.model_name + '_checkpoint_' + str(round_id) + '.t7'
        if round_id == 0:
            if not os.path.exists(checkpoint_name):
                torch.save(state, checkpoint_name)
                self.logging('save model to: ' + checkpoint_name)
        else:
            torch.save(state, "./NPY/" + checkpoint_name)
            self.logging('save model to: NPY/' + checkpoint_name)
            
    def get_flattened_parameter_tensor(self) -> torch.Tensor:
        tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for p in self.model.parameters():
            if p.requires_grad:
                flattened_param = p.data.reshape(-1)
                tensor_of_flattened_params = torch.cat((tensor_of_flattened_params, flattened_param),0)
        return tensor_of_flattened_params
    
    def get_param_shape(self):
        s_index = 0
        for param in self.model.parameters():
            if param.requires_grad:
                self.param_shape_list.append(param.data.shape)
                flatten_params = param.data.reshape(-1)
                e_index = s_index + len(flatten_params)
                self.cutoff_index_list.append((s_index, e_index))
                s_index = e_index

    def restore_model_with_tensor(self, received_tensor: torch.Tensor):
        i = 0
        for param in self.model.parameters():
            if param.requires_grad:
                param.data = received_tensor[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].reshape(self.param_shape_list[i])
                i += 1