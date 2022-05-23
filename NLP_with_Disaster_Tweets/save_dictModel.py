#The target of this page:生成词典，并保存为pkl文件，避免每次都重新生成

import pickle
from dataset_create import text_train_split
from build_vocab import word_Dict


ws = word_Dict()#初始化类


#统计词频到ws.count
for sentence in text_train_split :
    ws.fit(sentence)

# for idx , (key , value) in enumerate(ws.count.items()) :
#     print("idx:" , idx , "key:" , key , "value:" , value)

#生成词典ws.dict
ws.build_dict(min = 0 , max = 200 , max_features = 20000)
print(len(ws.dict))

#保存字典数据
pickle.dump(ws , open('./model/ws.pkl' , 'wb'))
