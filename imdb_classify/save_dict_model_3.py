"""
保存词库
"""
from tqdm import tqdm
from dict_2 import word_dict
import os
from imdbDataset_1 import tokenlize
import pickle

if __name__ == '__main__':
    ws = word_dict()
    train_data_path = r'D:\imdb\train'
    temp_path = [os.path.join(train_data_path , 'pos') , os.path.join(train_data_path , 'neg')]
    for path in temp_path :
        file_list = os.listdir(path)
        file_paths = [os.path.join(path , file) for file in file_list if file.endswith('.txt')]
        for file_path in tqdm(file_paths) :
            sentence = tokenlize(open(file_path , encoding = 'utf-8').read())
            ws.fit(sentence)#统计词频，生成count词典

    ws.build_dict(min = 10)#生成dict词典
    pickle.dump(ws , open('./model/ws.pkl' , 'wb'))
    print(len(ws))
