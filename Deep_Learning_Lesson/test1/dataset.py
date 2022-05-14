import torch
from torch.utils.data import TensorDataset , DataLoader
from get_data import getdata

train_data , test_data , train_labels , test_labels = getdata()#获取数据集

#将所有数据转为Tensor
train_data = torch.Tensor(train_data)
train_labels = torch.LongTensor(train_labels)
test_data = torch.Tensor(test_data)
test_labels = torch.LongTensor(test_labels)

def get_dataloader(train = True) :#构建数据加载器
    dataset = TensorDataset(train_data , train_labels) if train == True else TensorDataset(test_data , test_labels)#构造dataset
    dataloader = DataLoader(dataset = dataset , shuffle = True)#构造dataloader加载器，数据顺序打乱，batchsize默认
    return dataloader

#测试
# dataloader = get_dataloader(train = True)
# for idx , (train_data , train_label) in enumerate(dataloader) :
#     print(train_data , train_label)
#     break