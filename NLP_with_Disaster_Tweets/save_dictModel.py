#The target of this page:生成词典，并保存为pkl文件，避免每次都重新生成

import pickle
from dataset_create import text_train_split
from build_vocab import word_Dict


ws = word_Dict()#初始化类


for sentence in text_train_split :
    ws.fit(sentence)

ws.build_dict(min = 0 , max = 300)
print(len(ws))
