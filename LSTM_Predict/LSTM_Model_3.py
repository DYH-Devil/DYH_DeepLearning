"""
模型定义
"""

import torch.nn as nn
import lib

class LSTM_Model(nn.Module) :
    def __init__(self) :
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size = 1 , hidden_size = lib.hidden_size , num_layers = lib.num_layers , batch_first = True)
        self.fc = nn.Linear(in_features = lib.hidden_size , out_features = 1)
        #因为output[batchsize,seqlen,hiddensize * bidirectional],所以全连接层输入 in_features=hidden_size *bidiretional(1 or 2)

    def forward(self , input) :
        output , _ = self.lstm(input)
        output = output[: , -1 , :]#取sqlen上最后一个时间序列进行全连接
        output = self.fc(output)
        output = output.view(-1 , 1 , 1)
        return output