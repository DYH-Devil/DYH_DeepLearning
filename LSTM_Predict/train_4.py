"""
模型训练
"""
import numpy as np
import torch
import torch.nn as nn
from dataset_2 import get_dataloader
import lib
from LSTM_Model_3 import LSTM_Model
from torch.optim import Adam

lstm_model = LSTM_Model()#模型实例化
lstm_model = lstm_model.to(lib.device)
optimizer = Adam(lstm_model.parameters() , lr = 0.01)#优化器类实例化
loss_function = nn.MSELoss()#损失函数实例化


dataloader = get_dataloader(train = True)
def train(epoch) :
    for i in range(epoch):
        loss_list = []
        for idx, (input, target) in enumerate(dataloader):
            input = input.to(lib.device)
            target = input.to(lib.device)
            optimizer.zero_grad()#梯度归0
            predict = lstm_model(input)#预测
            loss = loss_function(predict , target)#计算损失
            loss_list.append(loss.item())
            loss.backward()#反向传播
            optimizer.step()#参数更新

            if idx % 5 == 0:
                torch.save(lstm_model.state_dict(), './model/model.pkl')  # 保存模型
                torch.save(optimizer.state_dict(), './model/optimizer.pkl')  # 保存优化器

        loss_mean = np.mean(loss_list)
        if i % 10 == 0 :
            print("epoch = {} | loss = {}".format(i , loss_mean))


if __name__ == '__main__':
    train(200)
