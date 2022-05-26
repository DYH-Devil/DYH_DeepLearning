"""
The target of this page:测试代码
"""

from torch.optim import  Adam
from dataset_create import x_test
import torch
from LSTM_Model import Twitter_BiLSTM
import config
import os
import pandas as pd

model = Twitter_BiLSTM(input_size = config.embedding_dim ,
                       hidden_size = config.hidden_size ,
                       num_layers = config.num_layers ,
                       drop_out = config.drop_out)
optimizer = Adam(model.parameters() , lr = 0.001)

#加载模型与优化器
if os.path.exists('./model/model.pkl') :
    model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))


#定义用于计算准确率的函数
def calc_accuracy(y_pred , y_true) :
    y_pred = torch.round(torch.sigmoid(y_pred))#取近似值
    correct = (y_pred == y_true).float()
    acc = correct.sum() / len(correct)#正确率
    return acc


def predict(x_test) :
    """
    :param x_test: 测试集数据:Tensor型
    :return target_list:预测结果:list型
    """
    target_list = []
    with torch.no_grad() :
        for text in x_test :
            text = text.unsqueeze(1).T
            y_pred = model(text)
            target = torch.round(torch.sigmoid(y_pred))#此时的target是一个Tensor
            #print(type(target))
            target = int(target.tolist()[0]) #转为list并取其中值
            target_list.append(target)
    return target_list

submissions = predict(x_test)
submission = pd.read_csv("./data/sample_submission.csv")
print(submission.info())
submission['target'] = submissions
submission.to_csv('./submission.csv',index=False)