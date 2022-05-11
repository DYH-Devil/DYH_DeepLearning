"""
完成对数据集的划分，dataloader制作
"""

import numpy as np
from feature_make import data
import lib
import torch
from torch.utils.data import TensorDataset , DataLoader
from feature_make import features

def create_dataset(data:list , time_step:int) :#划分特征集，标签集
    """
    :param data: 所使用的数据集
    :param time_step: 时间步
    :return:features特征集,labels标签集
    """
    features = []
    labels = []
    for i in range(len(data) - time_step - 1) :
        x = data[i : i + time_step , : ]#以前time_step的数据作为特征
        y = data[i + time_step , -1]#以第time_step的数据作为标签标签只取最后一个Close

        features.append(x)
        labels.append(y)

    return np.array(features) , np.array(labels) #将list转为array


def train_test_split(features:list , labels:list , train_size = lib.train_size) :#划分训练集，测试集
    """
    :param features: 上一步划分出的特征集
    :param labels: 上一步划分出的标签集
    :param train_size: 训练集比例
    :return: x_train , y_train , x_test , y_test
    """
    len_train = int(train_size * features.shape[0])#训练集数目
    x_train , y_train = features[:len_train , : , :] , labels[:len_train , : , :]
    x_test , y_test = features[len_train : , : , :] , labels[len_train : , : , :]

    return x_train ,y_train , x_test , y_test


def get_dataloader(train = True) :#制作dataloader数据加载器
    dataset = TensorDataset(x_train , y_train) if train == True else TensorDataset(x_test , y_test)
    dataloader = DataLoader(dataset = dataset , shuffle = True , batch_size = 4 , num_workers = 0)
    return dataloader


#测试
time_step = lib.time_step
data = features

features , labels = create_dataset(data , time_step)

features = torch.Tensor(features.reshape(-1 , time_step , 4)).to(lib.device)#把训练特征转为tensor放到GPU上
labels = torch.Tensor(labels.reshape(-1 , 1 , 1)).to(lib.device)#把训练标签转为tensor放到GPU上
#print("features_shape" , features.shape)#[248,3,4]
#print("labels_shape" , labels.shape)#[248,1,1]

x_train , y_train , x_test , y_test = train_test_split(features , labels , train_size = lib.train_size)
# print('x_train shape' , x_train.shape)
# print('y_train shape' , y_train.shape)
# print('x_test shape' , x_test.shape)
# print('y_test shape' , y_test.shape)
dataloader = get_dataloader(train = True)
for idx , (feature , label) in enumerate(dataloader) :
    print(features)
    print(label)
    break