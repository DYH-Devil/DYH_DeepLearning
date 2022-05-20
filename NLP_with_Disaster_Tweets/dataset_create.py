"""
The target of this page:构建dataset以及dataloader
"""

import torch
from torch.utils.data import TensorDataset , DataLoader
from text_clean import text_process , split_word
from get_data import text_train , text_test
from get_data import y_train as label_train


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

#构建dataset,dataloader
#-------------------------------------------------------------
# def creat_dataloader(x_train , y_train) :
#     dataset = TensorDataset(x_train , y_train)
#     dataloader = DataLoader(dataset = dataset , batch_size = 64 , shuffle = True)
#     return dataloader
#
# dataloader = creat_dataloader(text_train_split , label_train)
