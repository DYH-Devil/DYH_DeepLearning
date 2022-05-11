"""
模型训练
"""

import torch
from model import LSTM
import torch.nn as nn
from torch.optim import Adam
from dataset import get_dataloader
import os
import numpy as np
import lib

model = LSTM()#模型实例化
model = model.to(lib.device)
criterion = nn.MSELoss()#损失函数实例化
optimizer = Adam(model.parameters() , lr = 0.01)#优化器类实例化

if os.path.exists('./model/model.pkl') :
    model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

dataloader = get_dataloader(train = True)
def train(num_epoch) :
    for i in range(num_epoch) :
        loss_list = []
        for idx , (input , target) in enumerate(dataloader) :
            input = input.to(lib.device)
            target = target.to(lib.device)
            optimizer.zero_grad()  # 梯度置为0
            y_pred = model(input)
            loss = criterion(y_pred , target)#计算损失
            print("epoch:", i, "loss:", loss.item())
            loss_list.append(loss.item())
            loss.backward()#反向传播
            optimizer.step()#参数更新

            if idx % 5 == 0 :
                torch.save(model.state_dict() , './model/model.pkl')
                torch.save(optimizer.state_dict() , './model/optimizer.pkl')

        loss_mean = np.mean(loss_list)
        if i % 10 == 0 :
            print('epoch = {} | mean_loss = {} '.format(i , loss_mean))



if __name__ == '__main__':
    train(200)


