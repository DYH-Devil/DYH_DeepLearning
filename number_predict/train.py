import torch
import torch.nn.functional as F
from torch.optim import Adam
from dataset import dataloader
from seq2seq_model import seq2seq_Model
import config
import os

seq2seq_model = seq2seq_Model()
optimizer = Adam(seq2seq_model.parameters() , lr = 0.001)#优化器实例化

if os.path.exists('./model/model.pkl') :#模型加载
    seq2seq_model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

def train(epoch) :
    for i in range(epoch) :
        for idx , (input , label , input_len , label_len) in enumerate(dataloader) :
            # 梯度归0
            optimizer.zero_grad()
            #预测
            output , hidden = seq2seq_model(input , input_len , label , label_len)
            #print(output.size())
            #print(label.size())
            output = output.view(output.size(0) * output.size(1) , -1)
            label = label.view(-1)
            #计算损失
            loss = F.nll_loss(output , label , ignore_index=config.num_sequence.PAD)
            #反向传播
            loss.backward()
            #参数更新
            optimizer.step()
            print("idx:" , idx , "i:" , i , "loss:" , loss.item())

            if idx % 2 == 0 :#模型保存
                torch.save(seq2seq_model.state_dict() , './model/model.pkl')
                torch.save(optimizer.state_dict() , './model/optimizer.pkl')

if __name__ == '__main__':
    train(config.num_epochs)

