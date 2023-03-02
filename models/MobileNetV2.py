import torch.nn as nn
import torch.nn.functional as F
import torch
import torchsummary
import copy

def Conv3x3BNReLU(in_channels, out_channels, stride, groups, size):
    temp_size = copy.deepcopy(size)
    if stride == 2:
        temp_size[0] //= 2
        temp_size[1] //= 2
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
        nn.BatchNorm2d(out_channels),
        # nn.LayerNorm(temp_size, elementwise_affine=False),
        nn.ReLU6(inplace=True)
    )

def Conv1x1BNReLU(in_channels, out_channels, size):
    temp_size = copy.deepcopy(size)
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        # nn.LayerNorm(temp_size, elementwise_affine=False),
        nn.ReLU6(inplace=True)
    )

def Conv1x1BN(in_channels, out_channels, size):
    temp_size = copy.deepcopy(size)
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels)
        # nn.LayerNorm(temp_size, elementwise_affine=False)
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, size):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = (in_channels * expansion_factor)
        size1 = copy.deepcopy(size)
        size2 = copy.deepcopy(size)
        size3 = copy.deepcopy(size)
        if stride == 2:
            size3[0] //= 2
            size3[1] //= 2

        
        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_channels, mid_channels, size1),
            Conv3x3BNReLU(mid_channels, mid_channels, stride, groups=mid_channels, size=size2),
            Conv1x1BN(mid_channels, out_channels, size3)
        )
        

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels, size1)

    def forward(self, x):
        out = self.bottleneck(x)
        out = (out+self.shortcut(x)) if self.stride == 1 else out
        return out

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, t=6, size=[32,32]):
        super(MobileNetV2, self).__init__()

        self.first_conv = Conv3x3BNReLU(3,32,1,groups=1,size=size)
        size=[32,32]
        self.layer1 = self.make_layer(in_channels=32, out_channels=16, stride=1, factor=1, block_num=1,size=[32,32])
        size = [32,32]
        self.layer2 = self.make_layer(in_channels=16, out_channels=24, stride=2, factor=t, block_num=2, size=[32,32])
        size = [16,16]
        self.layer3 = self.make_layer(in_channels=24, out_channels=32, stride=2, factor=t, block_num=3, size=size)
        size = [8,8]
        self.layer4 = self.make_layer(in_channels=32, out_channels=64, stride=2, factor=t, block_num=4, size=size)
        size = [4,4]
        self.layer5 = self.make_layer(in_channels=64, out_channels=96, stride=1, factor=t, block_num=3, size=size)
        # self.layer6 = self.make_layer(in_channels=96, out_channels=160, stride=2, factor=t, block_num=3)
        # self.layer7 = self.make_layer(in_channels=160, out_channels=320, stride=1, factor=t, block_num=1)
        self.last_conv = Conv1x1BNReLU(96,160,[4,4])

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.linear = nn.Linear(in_features=160, out_features=num_classes)
        self.init_params()


    def make_layer(self, in_channels, out_channels, stride, factor, block_num, size):
        layers = []
        layers.append(InvertedResidual(in_channels, out_channels, factor, stride, size)) 
        if stride == 2:
            size[0] //= 2
            size[1] //= 2
            
        for i in range(1, block_num):
            layers.append(InvertedResidual(out_channels, out_channels, factor, 1, size))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = self.layer6(x)
        # x = self.layer7(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        x = self.linear(x)
        return x
    
def MobileNetV2_Cifar10():
    return MobileNetV2(10,6)

if __name__ == '__main__':
    model = MobileNetV2_Cifar10()
    torchsummary.summary(model,(3,32,32))
    input = torch.ones([1,3,32,32])
    output = model(input)
    print(output.size())
