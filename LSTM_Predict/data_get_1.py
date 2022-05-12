"""
完成对数据读取，可视化观察，以及归一化
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from plotly import graph_objects as go


df = pd.read_csv(r'D:/champagne_data/champagne.csv' , index_col = 0)#读取文件
print(df.head())

"""
1.数据可视化
"""

fig = go.Figure()#
fig.add_trace(go.Scatter(x = df.index , y = df['Sales'] , name = 'Sales'))
fig.show()

"""
2.数据归一化
"""
scaler = MinMaxScaler()
new_col_name = 'Sales_scaler'
df[new_col_name] = scaler.fit_transform(df['Sales'].values.reshape(-1 , 1))
print(df.head())