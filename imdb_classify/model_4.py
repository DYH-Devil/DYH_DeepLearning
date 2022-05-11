"""模型定义"""
#模型优化
#添加一个全连接层，使用激活函数
#把双向的LSTM的output再经过一个单向的LSTM在进行处理

import torch
import torch.nn as nn
from lib import ws , max_len , hidden_size , num_layers , bidirectional , dropout , device , test_batch_size
import torch.nn.functional as F


class MyModel(nn.Module) :
    def __init__(self):
        super(MyModel , self).__init__()
        self.embedding = nn.Embedding(len(ws) , embedding_dim = 100)#用100维向量表示一个词语
        """
        Embedding参数(词数 ， 维数)
        """
        #加入lstm
        self.lstm = nn.LSTM(input_size = 100 , hidden_size = hidden_size , num_layers = num_layers , batch_first = True , bidirectional = bidirectional , dropout = dropout)
        self.linear = nn.Linear(hidden_size * 2 , 64)
        self.output = nn.Linear(64 , 2)

    def forward(self , input):
        """
        :param input:[batchsize , seqlen(句子长度)]
        :return:
        """
        x = self.embedding(input)#x[batchsize , seqlen , 300]
        x , (h_n , c_n) = self.lstm(x)#x[batch_size , seqlen , bidirectional * hidden_size] h_n[num_layers * bidirectional , batch_size , hidden_size]
        #获取两个方向的最后一次output进行concat操作
        output_fw = h_n[-2 , : , :]#正向最后一次输出
        output_bw = h_n[-1 , : , :]#反向最后一次输出
        output = torch.cat([output_fw , output_bw] , dim = -1)#output[batch_size ,sqlen , hidden_size * 2]

        out = self.linear(output)

        out_relu = F.relu(out)#经过一次relu之后再进行全连接输出
        out = self.output(out_relu)

        return F.log_softmax(out , dim = -1)




