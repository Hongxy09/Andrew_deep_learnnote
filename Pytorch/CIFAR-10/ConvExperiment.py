'''
1.编写不同的卷积层（中心对齐和左上角对齐等等）
2.获得基向量图片
3.使用卷积层处理基向量图片
4.观察输出的卷积矩阵
'''
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
