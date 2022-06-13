#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2022 2022/6/6 8:17 下午
@Author  : keyoung
@File    : gcndemo2.py
"""
from networkx import to_numpy_matrix
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
zkc = nx.karate_club_graph()
order = sorted(list(zkc.nodes()))
A = to_numpy_matrix(zkc, nodelist=order) #领结矩阵

I = np.eye(zkc.number_of_nodes())
A_hat = A + I #增加闭环
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat)) #度矩阵

# 随机初始化权重
W_1 = np.random.normal(
    loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(
    loc=0, size=(W_1.shape[1], 2))

# 定义relu函数
def relu(X):
    return np.maximum(X, 0)
# 堆迭GCN层
def gcn_layer(A_hat, D_hat, X, W):
    return relu(D_hat**-1 * A_hat * X * W)

H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
output = H_2

def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum(1)
    return X_exp / partition  # 这里应用了广播机制

output = softmax(output)

feature_representations = {
    node: np.array(output)[node]
    for node in zkc.nodes()}



# nx.draw(zkc,node_size=300,with_labels = True,pos = nx.spring_layout(zkc),node_color = 'r')
# nx.draw(zkc)
# plt.show()