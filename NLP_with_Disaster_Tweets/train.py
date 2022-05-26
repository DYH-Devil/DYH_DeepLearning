"""
The target of this page:训练代码
"""

import torch
import config
from torch.optim import Adam
from LSTM_Model import Twitter_BiLSTM
from torch.nn import BCEWithLogitsLoss
from dataset_create import dataloader
import os

model = Twitter_BiLSTM(input_size = config.embedding_dim ,
                       hidden_size = config.hidden_size ,
                       num_layers = config.num_layers ,
                       drop_out = config.drop_out)

optimizer = Adam(model.parameters() , lr = 0.001)
criterion = BCEWithLogitsLoss()

model = model.to(config.device)

#加载模型与优化器类
if os.path.exists('./model/model.pkl') :
    model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

def train(num_epoch) :
    for i in range(num_epoch) :
        for idx , (x_train , y_train) in enumerate(dataloader) :
            x_train = x_train.to(config.device)
            y_train = y_train.to(config.device)

            #step1:梯度归0
            optimizer.zero_grad()
            #step2:计算
            out = model(x_train)
            #step3:计算损失
            loss = criterion(out , y_train)
            print("epoch:" , i , "idx:" , idx , "loss:" , loss.item())
            #step4:反向传播
            loss.backward()
            #step5:参数更新
            optimizer.step()
            if idx % 100 == 0:
                torch.save(model.state_dict(), './model/model.pkl')  # 保存模型
                torch.save(optimizer.state_dict(), './model/optimizer.pkl')  # 保存优化器

train(25)