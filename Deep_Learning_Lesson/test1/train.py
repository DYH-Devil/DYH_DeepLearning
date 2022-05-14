import torch
import torch.nn as nn
from Deep_Learning_Lesson.test1.dataset import get_dataloader
from Deep_Learning_Lesson.test1.model import iris_classificationModel
from torch.optim import Adam
import os

model = iris_classificationModel()

MSE = nn.CrossEntropyLoss()#交叉熵损失
optimizer = Adam(params = model.parameters() , lr = 0.001)#优化器使用Adam，学习率0.1


if os.path.exists('model/model.pkl') :
    model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))
    #加载模型与优化器类


def train(epoch) :
    """
    :param epoch: 训练次数
    :return:
    """
    dataloader = get_dataloader(train = True)#dataloader
    for i in range(epoch) :
        for idx , (train_data , train_label) in enumerate(dataloader) :
            optimizer.zero_grad()#梯度归0
            predict = model(train_data)#前向计算
            loss = MSE(predict , train_label)#计算损失
            print("epoch:", i, "loss:", loss.item())
            loss.backward()#反向传播
            optimizer.step()#更新参数

            if idx % 10 == 0:
                torch.save(model.state_dict(), 'model/model.pkl')  # 保存模型
                torch.save(optimizer.state_dict(), 'model/optimizer.pkl')  # 保存优化器




if __name__ == '__main__':
    train(100)
