import pandas as pd
import numpy as np

data_train = pd.read_csv('./data/train.csv')
data_test = pd.read_csv('./data/test.csv')

#标签
labels = data_train.target.to_numpy()
print(labels)


