import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
from tf_idf import corpus

newData = joblib.load('./model/pca.pkl')
kmeans = joblib.load('./model/model.pkl')


x , y = newData[: , 0] , newData[: , 1]

#获取label
y_pred = kmeans.labels_

#设置颜色
cluster_color = {0 : 'black' , 1 : 'red' , 2 : 'blue' , 3 : 'yellow' , 4 : 'green' }

#设置类名
cluser_names = {0 : '1类' , 1 : '2类' , 2 : '3类' , 3 : '4类' , 4 : '5类'}

df = pd.DataFrame(dict(x=x , y=y , label= y_pred , title = corpus))#构建一个dataframe
#print(df.head())
groups = df.groupby('label')
#print(groups.head())

fig , ax = plt.subplots(figsize = (8 , 5))
ax.margins(0.02)
for name , group in groups :
    ax.scatter(group.x , group.y , marker = 'o' ,  label = cluser_names[name] , color = cluster_color[name])

plt.show()

# labels = []
# # i = 0
# # while i < len(kmeans.labels_) :
# #     labels.append(kmeans.labels_[i])
# #     i += 1

