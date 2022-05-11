"""
训练流程
1.实例化模型，设置模型为训练模式
2.实例化优化器类，实例化损失函数
3.获取，遍历dataloader
4.梯度置为0
5.进行前向计算
6.计算损失
7.反向传播
8.更新参数
"""
import torch
import numpy as np
import torch.nn as nn
import torchvision
from torch.optim import Adam#优化器
import os
import torch.nn.functional as F

BATCH_SIZE = 128
TEST_BATCHSIZE = 1000
#1.准备数据集
def get_data(train = True , batchsize = BATCH_SIZE) :

    dataset = torchvision.datasets.MNIST('./data' , train = train , download = True ,#数据作为训练集
                                         transform = torchvision.transforms.Compose([#Compose中按顺序执行
                                             torchvision.transforms.ToTensor() ,#ToTensor转为tensor
                                             torchvision.transforms.Normalize((0.1307 , ),(0.3081 , ))#Nomalize正则化，分别传入均值mean和标准差std
                                         ]))

    #2.准备数据迭代器
    dataloader = torch.utils.data.DataLoader(dataset , batch_size = BATCH_SIZE , shuffle = True)#第一个参数dataset传入数据,第二个batchsize，第三个随机打乱
    return dataloader



#2.构建模型
class MNISTModel(nn.Module) :
    def __init__(self):
        super(MNISTModel,self).__init__()
        self.fc1 = nn.Linear(28*28 , 28)#第一次全连接，输入规格为28*28，经处理后输出规格为28
        self.fc2 = nn.Linear(28 , 10)#处理后分类为10个类型


    def forward(self , input):
        """
        input = [batch_size , 1 , 28 , 28]
        :param input:
        :return:
        """
        #1.修改形状
        x = input.view([input.size(0) , 1*28*28])#[1,28*28]
        #此时x被处理为28*28
        #2.全连接处理
        x = self.fc1(x)#对x全连接处理
        #3.激活函数处理
        x = F.relu(x)
        #4.输出处理，输出为10个类型
        out = self.fc2(x)

        return F.log_softmax(out , dim = -1)


#3.训练方法
def train(epoch) :
    dataloader = get_data()

    for idx , (input , target) in enumerate(dataloader) :
        output = model(input)#求预测值
        loss = F.nll_loss(output , target)#得到损失值

        optimizer.zero_grad()#梯度置为0
        loss.backward()#反向传播
        optimizer.step()#梯度更新

        if idx % 10 == 0 :
            print(epoch , idx , loss.item())

        if idx % 100 == 0 :
            torch.save(model.state_dict() , './model/model.pkl')
            torch.save(optimizer.state_dict(), './model/optimizer.pkl')
            #每间隔100次保存一次模型与优化器


def test() :
    loss_list = []
    acc_list = []
    test_dataloader = get_data(train = False , batchsize = TEST_BATCHSIZE)#训练集
    for idx , (input , target) in enumerate(test_dataloader) :
        with torch.no_grad():#测试不必对计算进行追踪
            output = model(input)
            cur_loss = F.nll_loss(output , target)
            loss_list.append(cur_loss)
            #计算准确率
            #output:[batch_size , 10]    target:[batch_size]

            #获取行方向上最大值的位置，意味着概率最大
            pred = output.max(dim = -1)[-1]
            cur_acc = pred.eq(target).float().mean()#准确率(单条数据)
            acc_list.append(cur_acc)

    print("平均准确率:" , np.mean(acc_list) , "平均损失:" , np.mean(loss_list))
#main
if __name__ == '__main__':
    model = MNISTModel()#实例化模型
    optimizer = Adam(model.parameters(), lr=0.001)#实例化优化器

    if os.path.exists('./model/model.pkl') :
        model.load_state_dict(torch.load('./model/model.pkl'))
        optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

    test()