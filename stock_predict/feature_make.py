import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler#用于特征值缩放
from plotly import graph_objects as go

file = r'D:/Data_stock/GOOG-year.csv'
data = pd.read_csv(file)

"""
数据可视化
"""
fig = go.Figure()
fig.add_trace(go.Scatter(x = data['Date'] , y = data['Close'] , name = 'Close Price'))
fig.show()



"""
特征归一化
"""
features = data[['Open' , 'High' , 'Low' , 'Close']]#选取四个特征
scaler = MinMaxScaler(feature_range = range(-1 , 1))#将特征缩放至(-1,1)
features = scaler.fit_transform(features.values.reshape(-1 , 4))
#print(features.shape)
