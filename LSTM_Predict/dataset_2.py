"""
完成对数据集的划分，dataloader制作
"""

import numpy as np
from data_get_1 import df
import lib
import torch
from torch.utils.data import TensorDataset , DataLoader


def create_dataset(data:list , time_step:int) :#划分出特征集和标签集
    """
    :param data: 所使用的数据集
    :param time_step: 时间步
    :return:features特征集,labels标签集
    """
    features , labels = [] , []
    for i in range(len(data) - time_step - 1) :
        x = data[i : i + time_step]#以前time_step的数据作为特征
        y = data[i + time_step]#以第time_step的数据作为标签

        features.append(x)
        labels.append(y)

    return np.array(features) , np.array(labels)#将list转为array


def train_test_split(features:list, labels:list , train_size = lib.train_size) :
    """
    :param features: 上一步划分出的特征集
    :param labels: 上一步划分出的标签集
    :param test_size: 测试集比例
    :return: x_train , y_train , x_test , y_test--------->训练集，测试集
    """
    len_train = int(features.shape[0] * train_size)
    x_train , y_train = features[: len_train , : , :] , labels[: len_train , : , :]#除了第一个维度其他均不变
    x_test , y_test = features[len_train : , : , :] , labels[len_train : , : , :]

    return x_train , y_train , x_test , y_test


def get_dataloader(train = True) :
    dataset = TensorDataset(x_train ,y_train) if train == True else TensorDataset(x_test , y_test)
    dataloader = DataLoader(dataset = dataset , batch_size = lib.batch_size , shuffle = True , num_workers = 0)
    return dataloader


#测试
time_step = lib.time_step
data =df['Sales_scaler']#data:series数据
features , labels = create_dataset(data.values , time_step)

#print(features.shape , labels.shape)
#features:(96,8) , labels(96,)

device = lib.device

features = torch.Tensor(features.reshape(-1 , time_step , 1)).to(device)#把训练特征转为tensor放到GPU上
labels = torch.Tensor(labels.reshape(-1 , 1 , 1)).to(device)#把训练标签转为tensor放到GPU上

#print(features.shape , labels.shape)
x_train , y_train , x_test , y_test = train_test_split(features , labels)

#print('x_train shape' , x_train.shape)
#print('y_train shape' , y_train.shape)
#print('x_test shape' , x_test.shape)
#print('y_test shape' , y_test.shape)

dataloader = get_dataloader(train = True)

for idx ,  (input , target) in enumerate(dataloader) :
    print(input.shape , target.shape)
    break
#查看第一个batch
#x , y = next(iter(dataloader))
#print(x.shape, y.shape)
#x.shape([10,8,1])----->LSTM中的input:[batchsize,sqlen,inputdim]
#y.shape([10,1,1])
