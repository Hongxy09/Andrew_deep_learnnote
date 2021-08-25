'''
CIFAR10神经网络，其实和MINST很像，可以说是一模一样
1.数据集导入CIFAR10
2.编写神经网络各层设计
3.导入训练数据，计算损失函数
4.选择更新策略进行参数更新
5.导入测试数据，计算准确度

程序应当实现的功能
1.训练网络（包括数据载入，数据训练，参数更新）
2.输出对比（绘制loss和acc曲线）
'''
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
global PATH
PATH = 'C:/code/Deeplearn_note/Pytorch/CIFAR-10/cifar_net.pth'
'''
1.导入数据集并规范化，部分图像可视化
'''
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 对输入数据进行规范化处理，将三通道的图像类型转化为张量并将范围归一化为[-1,1]
batch_size = 30
trainset = torchvision.datasets.CIFAR10(root='C:/code/SetData', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='C:/code/SetData', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')
# 图像输出
def imshow(img):
    img = img / 2 + 0.5  # unnormalize，将数据重新化为[0，1]方便显示图像
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # 交换输入的通道数据，因为img的格式是(channels,imgsize,imgsize)而imshow的格式是(imgsize,imgsize,channels)
    plt.show()
def print_img(number=1):
    '''
    打印图片的函数，number是打印几份batchsize数目的图。默认打印一份即4张图片
    '''
    dataiter = iter(trainloader)  # for
    total_images, total_labels = dataiter.next()
    if number > 1:
        for i in range(number-1):
            images, labels = dataiter.next()
            total_images = torch.cat((total_images, images), 0)
            total_labels = torch.cat((total_labels, labels), 0)
    imshow(torchvision.utils.make_grid(total_images))  # 拼图
    print(' '.join('%5s' % classes[total_labels[j]]
          for j in range(number*batch_size)))

'''
2.神经网络设计（输入层+隐藏层+输出层）
INPUT-conv->C1(6@28x28)-subsamling->S2(6@14x14)-conv->C3(16@10x10)-subsampling->S4(16@5x5)
    -flatten->C5(120)-linear->F6(84)-linear->F7(84)->OUTPUT
深度CNN过程=输入->卷积池化->卷积池化->全连接->全连接->输出
'''
class CIFARNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)  # 池化窗口2x2
        self.conv1 = nn.Conv2d(3, 6, 5)  # 输入通道数3输出通道数6卷积核5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # x输入包括了n，n不展平注意
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''
3.选择损失函数和更新策略
4.导入训练数据进行迭代更新
'''
def trainNet_once(net,loss_fn,optimizer,scheduler,device):
    '''
    训练神经网络，损失函数=交叉熵，权重更新策略=Momentum优化下的SGD
    只进行一次循环
    '''
    # net.load_state_dict(torch.load(PATH))#导入之前训练的数据
    scheduler.step()
    for i, data in enumerate(trainloader, 0):
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)
        '''
        此处inputs是四张图像的集合（batch_size=4）labels也是四个标签的集合
        在这个内层for中，先计算这4张图片的损失函数平均值
        当循环次数i到达2000时/计算完4*2000张图时就计算一次整体的损失函数（其实就是取2000次的平均值）
        torch.Size([4, 3, 32, 32])
        tensor([8, 6, 4, 0])
        '''
        optimizer.zero_grad()
        # 梯度计算+参数更新
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)  # 计算损失函数，对每一个都算损失函数
        loss.backward()
        optimizer.step()
    torch.save(net.state_dict(), PATH)
    return loss

def calculation_accuracy(net,device,loss_fn,eachclass=False,set='test'):
    '''
    5.导入测试数据，计算准确度
    如果计算的是当前整个网络预测的准确率则返回一个int值否则返回一个list
    '''
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    net.load_state_dict(torch.load(PATH))
    sum_correct_count = 0
    with torch.no_grad():
        #选择计算train集或者是test集的准确度
        funset =testloader if set=='test'else trainloader
        for data in funset:
            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)  # 通过神经网络预测类别
            loss = loss_fn(outputs, labels)  # 计算损失函数，对每一个都算损失函数
            _, predictions = torch.max(outputs, 1)  # 返回的是概率最大的值的下标
            for label, prediction in zip(labels, predictions):  # 返回由元组组成的列表
                if label == prediction:
                    correct_pred[classes[label]] += 1
                # label是图像对应的标签值，predication是预测的标签值int，如果两个值相等则在字典中该下标对应的类别下+1
                total_pred[classes[label]] += 1  # 记录预测过的该类图像数目
    #选择输出分类别的准确率或是整体的准确率
    if eachclass == True:
        acc_list=[]
        for classname, correct_count in correct_pred.items():
            accuracy_eachclass = 100 * float(correct_count) / total_pred[classname]
            print("Accuracy of eachclass {:5s} is: {:.1f} %".format(
                classname, accuracy_eachclass))
            acc_list.append(accuracy_eachclass)
        return acc_list
    else:
        sum_correct_count = sum(correct_pred.values())
        sum_total_count = sum(total_pred.values())
        accuracy=100 * sum_correct_count / sum_total_count
        print('Accuracy of %5s network: %d %%' % (set,accuracy))
        return accuracy,loss

def calculation_test_loss(net,loss_fn,device):
    with torch.no_grad():
        for i, data in enumerate(trainloader, 0):
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)  # 计算损失函数，对每一个都算损失函数
        return loss

def trainNet_mul(number):
    '''number=重复训练的次数'''
    net=CIFARNET()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #学习率衰减策略选择
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    #引入GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    #保存数据并打印图像
    train_acc_list=[]
    test_acc_list=[]
    train_loss_list=[]
    test_loss_list=[]
    train_loss,test_loss=0.0,0.0
    for epoch in range(number):
        train_loss=trainNet_once(net,loss_fn,optimizer,scheduler,device)
        test_loss=calculation_test_loss(net,loss_fn,device)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(calculation_accuracy(net,device,loss_fn,set='train'))
        test_acc_list.append(calculation_accuracy(net,device,loss_fn,set='test'))
        print(f"In epoch {epoch}:")
        print("loss of train = {:.5f} and test = {:.5f}".format(train_loss,test_loss))
    print(f"train loss and acc:{train_loss_list}\n{train_acc_list}\ntest loss and acc{test_loss_list}{test_acc_list}")    
    print("Finish!")

trainNet_mul(3)

