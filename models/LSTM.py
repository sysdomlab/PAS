import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
from tqdm import tqdm

class LSTM_KWS(nn.Module):

    def __init__(self):
        super(LSTM_KWS, self).__init__()
        self.rnn = nn.LSTM(
            # input_size=28,
            input_size=10,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
        self.out = nn.Linear(64,10)

    def forward(self,x):
        self.rnn.flatten_parameters()
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:,-1,:])
        return out


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
    

if __name__ == "__main__":
    model = LSTM_KWS()
    train_dataset = KWSconstructor(root=f'./kws/index_train.txt', transform=None)
    test_dataset = KWSconstructor(root=f'./kws/index_test.txt', transform=None)
    class_num = 10

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128)

    optimizer = SGD(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    device = torch.device("cuda:2")
    model = model.to(device)
    loss_func = loss_func.to(device)

    for epoch_num in range(1000):
        total_acc_train = 0
        total_loss_train = 0
        for inputs, labels in tqdm(train_dataloader, desc="Train"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            batch_loss = loss_func(outputs, labels)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            predict = torch.max(outputs, 1)[1].data
            acc = (predict == labels.data).sum()
            total_acc_train += acc
            total_loss_train += batch_loss.item()

        # ----------- 验证模型 -----------
        model.eval()
        total_acc_val = 0
        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for inputs, labels in tqdm(test_dataloader, desc="Test"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                batch_loss = loss_func(outputs, labels)
                predict = torch.max(outputs, 1)[1].data
                acc = (predict == labels.data).sum()
                total_acc_val += acc
                total_loss_val += batch_loss.item()
            
            print(f'''Epochs: {epoch_num + 1} 
            | Train Loss: {total_loss_train / len(train_dataset): .3f} 
            | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
            | Val Loss: {total_loss_val / len(test_dataset): .3f} 
            | Val Accuracy: {total_acc_val / len(test_dataset): .3f}''')
            
        model.train()
