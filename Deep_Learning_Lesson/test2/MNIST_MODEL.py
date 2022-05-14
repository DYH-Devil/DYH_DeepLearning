import torch.nn as nn
import torch.nn.functional as F


#2.构建模型
class MNISTModel(nn.Module) :
    def __init__(self):
        super(MNISTModel,self).__init__()
        self.fc1 = nn.Linear(28*28 , 28)#第一次全连接，输入规格为28*28，经处理后输出规格为28
        self.fc2 = nn.Linear(28 , 10)#处理后分类为10个类型(0-9)


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
        x = F.leaky_relu(x)
        #4.输出处理，输出为10个类型
        out = self.fc2(x)

        return F.log_softmax(out , dim = -1)
