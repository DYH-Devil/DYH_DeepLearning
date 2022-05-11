import torch
import torch.nn as nn
from dataset import get_dataloader
from model import iris_classificationModel
from torch.optim import Adam
import numpy as np
import os

model = iris_classificationModel()
MSE = nn.CrossEntropyLoss()#损失定义为交叉熵损失
optimizer = Adam(params = model.parameters() , lr = 0.001)#优化器使用Adam，学习率0.1

if os.path.exists('./model/model.pkl') :
    model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))
    #加载模型与优化器类

def accuracy(predict , label) :#计算准确率
    predict = torch.argmax(predict , dim = 1)
    acc = (predict == label).float().mean()
    return acc

def test() :
    loss_list = []
    acc_list = []
    test_loader = get_dataloader(train=False)
    for idx, (input, target) in enumerate(test_loader):
        with torch.no_grad():
            output = model(input)
            cur_loss = MSE(output, target)
            loss_list.append((cur_loss.cpu().item()))
            # 计算准确率
            pred = output.max(dim=-1)[1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.cpu().item())

    print("total loss,acc:", np.mean(loss_list), np.mean(acc_list))


if __name__ == '__main__':
    test()