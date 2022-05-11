"""
配置文件
"""

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

time_step = 8#时间步长

train_size = 0.8

batch_size = 10

hidden_size = 6#隐藏层单元数目

num_layers = 3#LSTM层数
