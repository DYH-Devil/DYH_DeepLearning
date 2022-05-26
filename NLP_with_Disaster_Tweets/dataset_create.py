"""
The target of this page:构建dataset以及dataloader
"""

import torch
from torch.utils.data import TensorDataset , DataLoader
from text_clean import text_process , split_word
from get_data import text_train , text_test
from get_data import y_train as label_train
from config import ws
import config

#清洗
#-------------------------------------------------------------
text_train_process = text_process(text_train)
#print(text_train_process)
text_test_process = text_process(text_test)
# print(text_test_process)
#-------------------------------------------------------------

#分词
#-------------------------------------------------------------
text_train_split = split_word(text_train_process)
text_test_split = split_word(text_test_process)

# print(text_train_split)
# print(text_test_split.shape)
#-------------------------------------------------------------

# 构建dataset,dataloader
#-------------------------------------------------------------
def creat_dataloader(x_train , y_train) :
    dataset = TensorDataset(x_train , y_train)
    dataloader = DataLoader(dataset = dataset , batch_size = 128 , shuffle = True)
    return dataloader

#step1:将分好词的句子转化为数字序列，并转为Tensor类型数据
x_train = [ws.transform(i , max_len = config.max_len) for i in text_train_split]
x_train =torch.LongTensor(x_train)
y_train = label_train
y_train = torch.Tensor(y_train)

x_test = [ws.transform(i , max_len = config.max_len) for i in text_test_split]#测试集转为数字序列
x_test = torch.LongTensor(x_test)

#step2:构建dataloader
dataloader = creat_dataloader(x_train , y_train)

# test
# for idx , (x_train , y_train) in enumerate(dataloader) :
#     print(x_train.shape , y_train.shape)
#     #x_train.shape[64,300] mean:[batchsize , seq_len]
#     break

#-------------------------------------------------------------