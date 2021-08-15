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
#基于minbatch计算，计算单个样本要用假minibatch即tensor.unsqueeze(n)
super(Net, self).__init__()#父类属性初始化
#卷积层的调用conv
#权重是滤波器
m = nn.Conv2d(16=输入通道数, 33=输出通道数, 5=滤波器尺寸, stride=(2, 1), padding=(4, 2))
input = torch.randn(20, 16, 50, 100)#N,C,H,W=输入数据的个数，通道数，高度，长度
output = m(input)
#全连接层的调用linear=Affine做了矩阵变换运算，注意一下权重矩阵下标和数学的矩阵下标有所差别，具体说则是前一层神经元的数目*后一层神经元的数目(此处的计算式是y=XW+B)
m = nn.Linear(20, 30)#W
input = torch.randn(128, 20)#X
output = m(input)
output.size = torch.Size([128, 30])#y
#池化层
torch.nn.MaxPool2d(kernel_size=池化窗口尺寸!=卷积核/滤波器尺寸, stride=None=默认等于池化窗口尺寸,如果剩下的数据不够滑动就丢弃, padding=0, dilation=1, return_indices=False, ceil_mode=False)#取窗口中的最大值
#展平数据
torch.flatten(input, start_dim=0, end_dim=-1) → Tensor
#选择损失函数
loss_fn=nn.MSELoss()
#选择梯度更新策略
import torch.optim as optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
# in your training loop:
optimizer.zero_grad() 
#计算损失函数梯度
output=net(input)
loss=loss_fn(nn(input), target)
loss.backward()
#更新参数
optimizer.step()    # Does the update

#实际操作中需要注意
#数据集导入
torch.utils.data.DataLoader(dataset=数据集来源, batch_size=1, shuffle=False=每个周期洗牌打乱, sampler=None=提取样本, batch_sampler=None=提取批量样本, num_workers=0=数据加载的子进程数, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=2, persistent_workers=False)
#图像输出
def imshow(img):
    img = img / 2 + 0.5# unnormalize，将数据重新化为[0，1]方便显示图像
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #交换输入的通道数据，因为img的格式是(channels,imgsize,imgsize)而imshow的格式是(imgsize,imgsize,channels)
    plt.show()
# get some random training images
dataiter = iter(trainloader)#for
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))#拼图
# make_grid的作用是将若干幅图像拼成一幅图像。
# 其中padding的作用就是子图像与子图像之间的pad有多宽。nrow是一行放入八个图片。
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
#join方法str.join(sequence)在sequence之间以str连接
#其中后面那个%是转义符
print("Accuracy for class {:5s} is: {:.1f} %".format(classes[2],2.12345))
#{:5s}是占位符，输出Accuracy for class frog  is: 2.1 %
running_loss += loss.item()
#item：取出单元素张量的元素值并返回该值，保持原元素类型不变，即：原张量元素为整形，则返回整形，原张量元素为浮点型则返回浮点型
a=np.random.rand(dim0,dim1...dimn)#0-1均匀分布随机数组
a=np.random.random(size=None)#浮点数
a=np.random.randint(low, high=None, size=None, dtype='I')  #整数     

```
6.  引入GPU计算
  ```python
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  net.to(device)
  inputs, labels = data[0].to(device), data[1].to(device)
  ```
