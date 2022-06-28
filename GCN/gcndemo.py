#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2022 2022/6/6 4:43 下午
@Author  : keyoung
@File    : gcndemo.py
"""

# from numpy import np  错误
import numpy as np #正确

# 定义邻接矩阵
A = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float
)

# 定义特征
X = np.matrix([
    [i, -i]
    for i in range(A.shape[0])
], dtype=float)

# f（A，X）
re1 = A * X

# 增加自己与自己的链接
I = np.matrix(np.eye(A.shape[0])) #单位阵
A_hat = A + I
re2 = A_hat * X

# 计算度矩阵
D = np.array(np.sum(A, axis=0))[0]
D = np.matrix(np.diag(D))

# 增加了自环以后的度矩阵
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

# 定义应用权重 2*2 维的
W = np.matrix([
    [1, -1],
    [-1, 1]
])
re3 = D_hat**-1 * A_hat * X * W

# 如果需要减小输出维度  则减小W维度即可
# Wt = np.matrix([
#     [1],
#     [-1]
# ])
# re3t = D_hat**-1 * A_hat * X * W
#
# def relu(x):
#     x[x <= 0] = 0
#     x[x > 0] = 1
#     return x

# 增加激活函数后
def relu(X):
    return np.maximum(X, 0)

re4 = relu(re3)
