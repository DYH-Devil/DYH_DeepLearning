"""
The target of this page:构建双向LSTM模型
"""

import torch
import torch.nn as nn
import config

class Twitter_BiLSTM(nn.Module) :
    def __init__(self , input_size , hidden_size , num_layers , drop_out):
        super(Twitter_BiLSTM, self).__init__()

        self.embedding = nn.Embedding(len(config.ws.dict) ,
                                      embedding_dim = config.embedding_dim ,
                                      )#词嵌入，将词转为向量[vocab_size , embedding_dim]

        self.BiLSTM = nn.LSTM(input_size = input_size ,
                              hidden_size = hidden_size ,
                              batch_first = True ,
                              bidirectional = True ,
                              num_layers = num_layers ,
                              dropout = drop_out
                              )

        self.fc1 = nn.Linear(hidden_size * 2 , hidden_size)
        self.fc2 = nn.Linear(hidden_size , 1)
        self.drop = nn.Dropout(0.2)

    def forward(self , text):
        embedded = self.embedding(text)# embedded:[batch_size , seq_len , embedding_dim]
        output , (h_n , c_n) = self.BiLSTM(embedded)
        #output:[batch_size , seq_len , hidden_size * bidirectional]
        #h_n , c_n:[num_layers * bidirectional , batch_size , hidden_size]

        #取两个方向上最后一次output进行concat操作
        output_fw = h_n[-2, :, :]  # 正向最后一次输出
        output_bw = h_n[-1, :, :]  # 反向最后一次输出
        out = torch.cat([output_fw, output_bw], dim=-1)  # output[batch_size ,hidden_size * 2]
        out = self.fc1(out)
        out = self.drop(self.fc2(out))
        return out.squeeze(1)
