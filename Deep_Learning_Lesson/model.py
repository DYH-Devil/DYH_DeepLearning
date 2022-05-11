import torch.nn as nn

class iris_classificationModel(nn.Module) :
    def __init__(self):
        super(iris_classificationModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4 , 32) , #第一层网络:input4 --> output:32
            nn.ReLU() , #使用ReLu激活函数
            nn.Linear(32 , 3) , #30 --> 3完成分类
        )

    def forward(self , input):
        output = self.fc(input)
        return output
