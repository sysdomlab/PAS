import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import copy

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

def ResNet18_Cifar10():
    return ResNet(ResidualBlock)

if __name__ == '__main__':
    model = ResNet18_Cifar10()
    torchsummary.summary(model,(3,32,32))
    input = torch.ones([1,3,32,32])
    output = model(input)
    print(output.size())