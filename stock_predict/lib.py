"""
配置文件
"""
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hidden_size = 32
num_layers = 2
input_size = 4
output_size = 1
num_epoch = 100
train_size = 0.8
time_step = 3