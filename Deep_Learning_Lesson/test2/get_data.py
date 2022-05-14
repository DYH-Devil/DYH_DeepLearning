import torchvision
import torch
import config


#准备数据集
def get_data(train = True , batchsize = config.BATCH_SIZE) :

    #1.构建数据集dataset
    dataset = torchvision.datasets.MNIST('./data' , train = train , download = True ,#数据作为训练集
                                         transform = torchvision.transforms.Compose([#Compose中按顺序执行
                                             torchvision.transforms.ToTensor() ,#ToTensor转为Tensor
                                             torchvision.transforms.Normalize((0.1307 , ),(0.3081 , ))#Nomalize正则化，分别传入均值mean和标准差std
                                         ]))

    #2.准备dataloader数据迭代器
    dataloader = torch.utils.data.DataLoader(dataset , batch_size = config.BATCH_SIZE , shuffle = True)#数据集为dataset,打乱，batch_size默认为训练batch:128
    return dataloader


