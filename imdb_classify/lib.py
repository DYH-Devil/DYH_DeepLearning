"""
配置文件存放，需要的时候直接从lib中导入即可
"""
import pickle
import torch

ws = pickle.load(open('./model/ws.pkl' , 'rb'))

max_len = 200#句子最大词数

batch_size = 128

test_batch_size = 64

hidden_size = 128#每个隐藏层有128单元

num_layers = 2#2层隐藏层

bidirectional = True#双向lstm

dropout = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")