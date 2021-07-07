# import tensorflow as tf
# print("tensorflow version:",tf.__version__)
# # C:/Otherapp/Anaconda/envs/tensorflowenvs/python.exe
import numpy as np
Q = np.mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
D = np.diag([10, 100, 1000])
P = Q.T

q = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
p = q.T
d = np.array([10, 100, 1000])

a = np.dot(np.dot(Q, D), P)
b = 0
for i in range(3):
    b += d[i]*np.dot(q[i], p[i])
print(a)
print(b)
print(a)
