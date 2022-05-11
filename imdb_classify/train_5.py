"""训练定义"""
import torch

from lib import batch_size , device
from imdbDataset_1 import dataLoader
from model_4 import MyModel
from torch.optim import Adam#优化器
import torch.nn.functional as F
import os
from tqdm import tqdm

imdb_model = MyModel()#实例化
imdb_model = imdb_model.to(device)#放在GPU上跑

optimizer = Adam(imdb_model.parameters() , lr = 0.01)#优化器类实例化

if os.path.exists('./model/model.pkl') :
    imdb_model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))
    #加载模型与优化器类


def train(epoch) :
    """
    :param epoch: 次数
    :return:
    训练流程
    1.梯度归0
    2.预测
    3.计算损失
    4.前向计算
    5.参数更新
    """
    dataloader = dataLoader(train=True)
    for idx , (input , target) in enumerate(dataloader) :
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()#梯度归0
        output = imdb_model(input)#预测
        loss = F.nll_loss(output , target)#计算损失
        print(idx, loss.item())  # 打印损失
        loss.backward()#反向传播
        optimizer.step()#参数更新

        if idx % 100 == 0 :
            torch.save(imdb_model.state_dict() , './model/model.pkl')#保存模型
            torch.save(optimizer.state_dict() , './model/optimizer.pkl')#保存优化器


if __name__ == '__main__':
    for i in range(10) :
        train(i)
