'''
1.编写不同的卷积层（中心对齐和左上角对齐等等）
2.获得基向量图片
3.使用卷积层处理基向量图片
4.观察输出的卷积矩阵
'''
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 对输入数据进行规范化处理，将三通道的图像类型转化为张量并将范围归一化为[-1,1]
batch_size = 30
# trainset = torchvision.datasets.CIFAR10(root='C:/code/SetData', train=True, download=False, transform=transform)
trainset = torchvision.datasets.CIFAR10(root='C:/Users/17853/code/SetData', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
# testset = torchvision.datasets.CIFAR10(root='C:/code/SetData', train=False, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='C:/Users/17853/code/SetData', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # 反规范化（unormalize）
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 默认 `log_dir` 是 "runs" —— 在这里我们更具体一些
writer = SummaryWriter("runs/convnet_1")
writer.add_scalar('pic', scalar_value, global_step=None, walltime=None)
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 创建图像网格（grid of images）
img_grid = torchvision.utils.make_grid(images)

# 显示图像
matplotlib_imshow(img_grid, one_channel=True)

# 写入到 TensorBoard
writer.add_image('four_fashion_mnist_images', img_grid)