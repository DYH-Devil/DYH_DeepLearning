import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D#3D图
from sklearn.neural_network import MLPClassifier#BP分类

from sklearn.datasets._samples_generator import make_classification#数据制作方法
from sklearn.metrics import classification_report , confusion_matrix
#评估方法

import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning
import pickle
from data_make import train_samples , train_labels

#step2:模型训练
BP = MLPClassifier(solver = 'sgd' ,
                   activation = 'relu' ,
                   max_iter = 50 ,
                   alpha = 1e-3 ,
                   hidden_layer_sizes = (32 , 32) ,
                   random_state = 1)
#参数解释:
#solver:优化器
#activacation:激活函数
#max_iter:迭代次数
#alpha:惩罚项
#hidden_layer_sizer:(神经层数 ， 单层神经元数目)

#查看模型参数
print(BP)

#模型训练
BP.fit(train_samples , train_labels)
pickle.dump(BP , open('./model/model.dat' , 'wb'))#保存模型


#step3:结果可视化
#(1)显示分类散点图
predict_label = BP.predict(train_samples)
fig = plt.figure()
ax = Axes3D(fig , rect = [0,0,1,1] , elev = 20 , azim = 20)
ax.scatter(train_samples[: , 0] ,
           train_samples[: , 1] ,
           train_samples[: , 2] ,
           marker = 'o' ,
           c = predict_label)
plt.title("Demo Data Predict With BP Model")
plt.show()

#(2)计算模型分数
model_score = BP.score(train_samples , train_labels)
print(model_score)
#准确率报表
print(classification_report(train_labels ,predict_label))
#混淆矩阵
classes = [0,1,2,3]
print(confusion_matrix(train_labels , predict_label  , classes = classes))