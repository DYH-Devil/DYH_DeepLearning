import torch
from torch.utils.data import DataLoader , Dataset
import numpy as np
import config

def collate_fn(batch) :
    batch = sorted(batch , key = lambda x : x[3] , reverse = True)#按label长度降序排列
    input , label  , input_len , label_len = zip(*batch)
    input = torch.LongTensor([config.num_sequence.transform(i, max_len=config.max_len) for i in input])
    label = torch.LongTensor([config.num_sequence.transform(i, config.max_len + 1 , addEoS = True) for i in label])
    input_len = torch.LongTensor(input_len)
    label_len = torch.LongTensor(label_len)
    return input, label , input_len , label_len



class Number_dataset(Dataset) :
    def __init__(self):
        np.random.seed(10)#由于数据随机生成，因此需要设置随机种子防止其改变
        self.train_data = np.random.randint(0 , 1e8 , size = [50000])

    def __getitem__(self, index):
        input = list(str(self.train_data.data[index]))
        label = input + ['0']
        input_len = len(input)
        label_len = len(label)
        return input , label , input_len , label_len

    def __len__(self):
        return self.train_data.data.shape[0]


dataloader = DataLoader(dataset = Number_dataset() , batch_size = config.train_batch , shuffle = True , collate_fn = collate_fn)
for idx  , (input , label , input_len , label_len) in enumerate(dataloader) :
    print("input_size :" , input.size())
    print("label size :" , label.size())
    break
