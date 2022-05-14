import torch
from get_data import get_data
from MNIST_MODEL import MNISTModel
import torch.nn.functional as F
from torch.optim import  SGD
import os
import config
import numpy as np

model = MNISTModel()#实例化模型
optimizer = SGD(model.parameters() , lr = 0.001)#实例化优化器，传入模型参数，学习率

if os.path.exists('./model/model.pkl') :#加载模型与优化器
    model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

dataloader = get_data(train=True)#训练dataloader,train置为True

#模型训练
def train(epoch) :
    loss_list = []  # 存loss
    acc_list = []  # 存acc
    for i in range(epoch) :
        for idx , (input , target) in enumerate(dataloader) :
            optimizer.zero_grad()  # 梯度置为0
            output = model(input)#求预测值
            loss = F.nll_loss(output , target)#求损失值
            loss_list.append(loss.detach().numpy())
            loss.backward()#反向传播
            optimizer.step()#梯度更新

            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()  # 准确率(单条数据)
            acc_list.append(cur_acc)

            if idx % 100 == 0 :
                print("epoch:" , i , "idx:" , idx , "loss:" , loss.item())#打印损失

            if idx % 100 == 0 :
                torch.save(model.state_dict() , './model/model.pkl')
                torch.save(optimizer.state_dict(), './model/optimizer.pkl')
                #每间隔100次保存一次模型与优化器
    print("在训练集上平均准确率:", np.mean(acc_list), "平均损失:", np.mean(loss_list))


if __name__ == '__main__':
    train(config.num_epoch)