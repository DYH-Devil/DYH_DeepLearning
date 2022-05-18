"""
The target of this page:读取数据，划分训练集与测试集,标签，特征
"""

import pandas as pd

data_train = pd.read_csv('./data/train.csv')
data_test = pd.read_csv('./data/test.csv')

# print(data_train.info())
# print(data_test.info())

#训练集特征
x_train = data_train.drop(['id' , 'keyword' , 'location' , 'target'] , axis = 1)

#训练集标签
y_train = data_train.target.to_numpy().reshape(-1)

#测试集标签
x_test = data_test.drop(['id' , 'keyword' , 'location'] , axis = 1)

#print(x_train.iloc[1])

# print("x_train:" , x_train)
# print("y_train:" , y_train)
# print("x_test:" , x_test)


