import torch
import torch.nn as nn
import  lib

#print(type(x_train))
"""
LSTM定义模型
"""
class LSTM(nn.Module) :
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size = lib.input_size , hidden_size = lib.hidden_size ,
                            num_layers = lib.num_layers , batch_first = True)#暂时只用单向LSTM

        self.linear = nn.Linear(lib.hidden_size , lib.output_size)

    def forward(self , input):
        """
        :param input:(batchsize,sqlen,inputsize)
        :return:
        """
        #h_0 = torch.zeros(lib.num_layers , input.size(0) , lib.hidden_size).requires_grad_()
        #c_0 = torch.zeros(lib.num_layers , input.size(0) , lib.hidden_size).requires_grad_()
        #初始化h_0和c_0
        #c_0,h_0,c_n,h_n:[num_layers*bidirectional,batchsize,hiddensize]

        output , (h_n , c_n) = self.lstm(input)
        output = self.linear(output[: , -1 , :])#output:[batchsize,seqlen,num_directional*hiddensize]
        #在最后一个sqlen上进行全连接
        output = output.view(-1 , 1 , 1)
        return output