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