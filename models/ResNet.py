import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import copy
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, size=[]):
        super(ResidualBlock, self).__init__()
        size1 = copy.deepcopy(size)
        if stride == 2:
            size1[0] //= 2
            size1[1] //= 2
        size2 = copy.deepcopy(size1)
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.BatchNorm2d(outchannel),
            nn.LayerNorm(size1, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(outchannel)
            nn.LayerNorm(size2, elementwise_affine=False)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(outchannel)
                nn.LayerNorm(size1, elementwise_affine=False)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LayerNorm([32,32], elementwise_affine=False),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1, size=[32,32])
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2, size=[32,32])
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2, size=[16,16])
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2, size=[8,8])
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride, size):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, size))
            if stride == 2:
                size[0] //= 2
                size[1] //= 2
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # print('forward')
        return out

class ResNet4Cifar(nn.Module):
    def __init__(self, block, num_block, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(16),
            nn.LayerNorm([32, 32], elementwise_affine=False),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 16, 3, 1, [32,32])
        self.conv3_x = self._make_layer(block, 32, 3, 2, [32,32])
        self.conv4_x = self._make_layer(block, 64, 3, 2, [16,16])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, size))
            if stride == 2:
                size[0] //= 2
                size[1] //= 2
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

def ResNet20_Cifar10(num_classes=10, **kargs):
    """ return a ResNet 20 object
    """
    return ResNet4Cifar(ResidualBlock, [3, 3, 3], num_classes=num_classes)


def ResNet18_Cifar10():
    return ResNet(ResidualBlock, num_classes=10)
    # return ptcv_get_model("resnet20_cifar10")

if __name__ == '__main__':
    model = ResNet20_Cifar10()
    
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.1
    device = torch.device('cuda:1')
    
    train_dataset = datasets.CIFAR10(root='/home/zgan/Datasets/Cifar10', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10(root='/home/zgan/Datasets/Cifar10', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    model.to(device)
    
    # 微调ViT模型
    for epoch in range(num_epochs):
        print("epoch:",epoch)
        model.train()
        train_loss = 0.0
        train_correct = 0
    
        bar = tqdm(train_loader, desc="Train")
        for images, labels in bar:
            images = images.to(device)
            labels = labels.to(device)
    
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
    
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
    
        # 在训练集上计算准确率
        train_accuracy = 100.0 * train_correct / len(train_dataset)
    
        # 在测试集上进行评估
        model.eval()
        test_loss = 0.0
        test_correct = 0
    
        with torch.no_grad():
            bar = tqdm(test_loader, desc="Test")
            for images, labels in bar:
                images = images.to(device)
                labels = labels.to(device)
    
                outputs = model(images)
                loss = criterion(outputs, labels)
    
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
    
        # 在测试集上计算准确率
        test_accuracy = 100.0 * test_correct / len(test_dataset)
    
        # 打印每个epoch的训练损失、训练准确率和测试准确率
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, '
              f'Test Accuracy: {test_accuracy:.2f}%')
    
    
    
    