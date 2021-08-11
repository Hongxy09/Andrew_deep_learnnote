## 安装步骤
* 参考https://blog.csdn.net/qq_45704942/article/details/114647667
1. 新建环境——在anaconda中新建一个环境python版本3.6
2. 安装——激活环境后在https://pytorch.org/get-started/locally/选择对应的版本，输入安装语句即可
3. 验证——prompt中输入`import torch`和`import torchvision`
   1. 写代码——在VSC中写一个test.py并快捷运行/或者在prompt中分别输入`cd../..`和`Users\17853\code\Deeplearn_note\Pytorch\test.py`
4. 验证错误——上一步前半部分成功，万事大吉开始学习；前半部分失败而后者成功那就是vsc没有切换环境，重新切换环境ctrl+p,>python:select...选择新建的环境解释器即可

5. 手册缩略版
```python
import torch
import numpy as np
import torchvision
#记录一些快捷指令

#新建张量
x_empty = torch.empty(5,3)#5*3元素未初始化的张量
x_rand = torch.rand(5,3)#5*3元素为随机数的张量
x_zeros=torch.zeros(5,3,dtype=torch.long)#5*3元素为0的张量
x_randperm=torch.randperm(n,dtype=torch.float32)#n个0~n-1整数的随机排列数列
x_randintmatrix=torch.randint(0,4,(3,4))#n个0~n-1整数的随机排列的张量
x_randn=torch.randn(2,3) #均值为0，方差为1

#从已有数据中获得张量
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)#手动输入数据
np_array = np.array(data)
x_np = torch.from_numpy(np_array)#从numpy数组中获得数据
x_ones = torch.ones_like(x_data)#从张量获得张量，保留属性不保留具体数据数据全部变成1
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float)#从张量获得张量，不保留属性不保留具体数据
print(f"Random Tensor: \n {x_rand} \n")#format的简化写法，{}里面是变量
x_select=torch.multinomial(input, num_samples,replacement=False, out=None)
#作用是对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标，replacement是是否放回
x_view = x_data.view(-1, 8)#改变大小：如果你想改变一个 tensor 的大小或者形状，你可以使用 torch.view:
a = torch.ones(5,5)#5*5全是1的张量

#张量的属性
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")#cpu

#张量运算

#加法
result = torch.empty(5, 3)
torch.add(x, y, out=result)#result输出加法
y.add_(x)#adds x to y替换加法，结果在y上

#维度相关
#张量的维度=阶数=轴，[[1,2],[3,4]]，是一个2阶张量，有两个维度或轴，从第一个维度=轴看到的是[1,3]，[2,4]两个向量
tensor.squeeze()#消除只有一个元素的维度（就是某个维度上这个张量是一个整体，在这个维度上再套一层，这个新层就是0维度）
tensor.unsqueeze(n)#在n维添加一个0行元素

#乘法
tensor_mul=tensor.matmul(tensor.T)
tensor_mul2=tensor @ tensor.T

#关于自动计算梯度
import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
prediction = model(data) # forward pass
loss = (prediction - labels).sum()
loss.backward() # backward pass
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step() #gradient descent
#冻结参数
from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False
#冻结除了最后一层的其他参数，即计算梯度的唯一参数是最后一层分类器的权重和偏置
#神经网络nn组织
super(Net, self).__init__()#父类属性初始化
#卷积层的调用conv
m = nn.Conv2d(16=输入通道数, 33=输出通道数, 5=滤波器尺寸, stride=(2, 1), padding=(4, 2))
input = torch.randn(20, 16, 50, 100)#N,C,H,W=输入数据的个数，通道数，高度，长度
output = m(input)
#全连接层的调用linear=Affine做了矩阵变换运算，注意一下权重矩阵下标和数学的矩阵下标有所差别，具体说则是前一层神经元的数目*后一层神经元的数目(此处的计算式是y=XW+B)
m = nn.Linear(20, 30)#W
input = torch.randn(128, 20)#X
output.size = torch.Size([128, 30])#y

```