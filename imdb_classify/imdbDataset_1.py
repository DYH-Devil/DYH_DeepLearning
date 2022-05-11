import os.path

import torch
from torch.utils.data import DataLoader , Dataset
import re
from lib import ws , max_len , batch_size , test_batch_size

def tokenlize(content) :#分词实现
    content = re.sub('<.*?>' , ' ' , content)
    filters = ['\t' , '\n' , '\x97' , '\x96' , '@' , '#' , '$' , '%' , '&']
    content = re.sub('|'.join(filters) , ' ' , content)#|表示综合所有的filters,全部替换
    tokens = [word.strip().lower() for word in content.split()]
    return tokens


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    texts , labels = list(zip(*batch))
    #labels = torch.tensor(batch[0], dtype=torch.int32)
    #texts = batch[1]
    texts_seq = [ws.transform(i , max_len = max_len) for i in texts]
    #del batch
    labels = torch.LongTensor(labels)
    texts_seq = torch.LongTensor(texts_seq)
    return texts_seq , labels#将两个返回值置为张量

class imdbDataset(Dataset) :
    def __init__(self , train = True):
        super(imdbDataset, self).__init__()
        self.train_data_path = r'D:\imdb\train'
        self.test_data_path = r'D:\imdb\test'

        data_path = self.train_data_path if train == True else self.test_data_path

        self.total_path = []#总路径
        temp_path = [os.path.join(data_path , 'pos') , os.path.join(data_path , 'neg')]#把neg和pos集中到同一个文件夹下
        for file_path in temp_path :#....pos    ....neg_
            file_list = os.listdir(file_path)#列表所有路径下的文件
            file_path_list = [os.path.join(file_path , i) for i in file_list if i.endswith('.txt')]
            self.total_path.extend(file_path_list)


            #for file in file_list :
                #if file.endswith('.txt') :
                    #file_path = [os.path.join(file_path , file)]
                    #self.total_path.extend(file_path)


    def __getitem__(self, index : int):#获取文本及标签
        path = self.total_path[index]

        label_str = path.split('\\')[-2]
        label = 1 if label_str == 'pos' else 0


        content = open(path , 'r' , encoding = 'utf-8').read()
        tokens = tokenlize(content)

        return tokens , label


    def __len__(self):
        return len(self.total_path)


def dataLoader(train = True , batch_size = batch_size) :#将dataloader定义为方法的原因主要是:在dataset中需要传入train，根据train值来确定是训练还是测试
    dataset = imdbDataset(train = train)
    data_Loader = DataLoader(dataset = dataset , batch_size = batch_size , shuffle = True , collate_fn = collate_fn , num_workers = 0)
    return  data_Loader


if __name__ == '__main__':
    #print(dataset[4])
    data = dataLoader(train = True)
    for idx , (label , content) in enumerate(data) :
        print(idx)
        print(label)
        print(content)
        break
