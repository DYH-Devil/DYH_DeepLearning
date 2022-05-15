import torch
from get_data import get_data
from MNIST_MODEL import MNISTModel
import torch.nn.functional as F
from torch.optim import  Adagrad
import os
import config
import numpy as np

model = MNISTModel()#实例化模型
optimizer = Adagrad(model.parameters() , lr = 0.001)#实例化优化器，传入模型参数，学习率
print(optimizer)

if os.path.exists('./model/model.pkl') :#加载模型与优化器
    model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))

test_dataloader = get_data(train = False , batchsize = config.TEST_BATCHSIZE)#测试集dataloader,因此train置为false，batchsize可设置大一些，为512
#测试
def test() :
    loss_list = []#存loss
    acc_list = []#存acc
    for idx , (input , target) in enumerate(test_dataloader) :
        with torch.no_grad():#测试不必对梯度进行追踪
            output = model(input)
            cur_loss = F.nll_loss(output , target)
            loss_list.append(cur_loss)
            #计算准确率
            #output:[batch_size , 10]    target:[batch_size]

            #获取行方向上最大值的位置，意味着概率最大
            pred = output.max(dim = -1)[-1]
            cur_acc = pred.eq(target).float().mean()#准确率(单条数据)
            acc_list.append(cur_acc)

    print("在测试集平均准确率:" , np.mean(acc_list) , "平均损失:" , np.mean(loss_list))


if __name__ == '__main__':
    test()