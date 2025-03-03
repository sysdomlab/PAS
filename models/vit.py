import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from transformers import ViTModel, ViTForImageClassification
from tqdm import tqdm
 
# 设置随机种子
torch.manual_seed(42)
 
# 定义超参数
batch_size = 32
num_epochs = 20
learning_rate = 0.1
device = torch.device('cuda:0')
 
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
 
# 加载CIFAR-10数据集
train_dataset = CIFAR10(root='/home/zgan/Datasets/Cifar10', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='/home/zgan/Datasets/Cifar10', train=False, download=True, transform=transform)
 
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
# 加载预训练的ViT模型
vit_model = ViTForImageClassification.from_pretrained('/home/zgan/Datasets/google/vit-base-patch16-224').to(device)
 
# 替换分类头
num_classes = 10
vit_model.config.classifier = 'mlp'
vit_model.config.num_labels = num_classes
vit_model.classifier = nn.Linear(vit_model.config.hidden_size, num_classes).to(device)
 
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vit_model.parameters(), lr=learning_rate)
 
# 微调ViT模型
for epoch in range(num_epochs):
    print("epoch:",epoch)
    vit_model.train()
    train_loss = 0.0
    train_correct = 0
 
    bar = tqdm(train_loader, desc="Train")
    for images, labels in bar:
        images = images.to(device)
        labels = labels.to(device)
 
        # 前向传播
        outputs = vit_model(images)
        loss = criterion(outputs.logits, labels)
 
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        train_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        train_correct += (predicted == labels).sum().item()
 
    # 在训练集上计算准确率
    train_accuracy = 100.0 * train_correct / len(train_dataset)
 
    # 在测试集上进行评估
    vit_model.eval()
    test_loss = 0.0
    test_correct = 0
 
    with torch.no_grad():
        bar = tqdm(test_loader, desc="Test")
        for images, labels in bar:
            images = images.to(device)
            labels = labels.to(device)
 
            outputs = vit_model(images)
            loss = criterion(outputs.logits, labels)
 
            test_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            test_correct += (predicted == labels).sum().item()
 
    # 在测试集上计算准确率
    test_accuracy = 100.0 * test_correct / len(test_dataset)
 
    # 打印每个epoch的训练损失、训练准确率和测试准确率
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')