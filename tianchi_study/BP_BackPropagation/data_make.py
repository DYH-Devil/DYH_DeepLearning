import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D#3D图
from sklearn.neural_network import MLPClassifier#BP分类

from sklearn.datasets._samples_generator import make_classification#数据制作方法
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.model_selection import train_test_split
#评估方法

import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning

#step1
#制作数据集
samples , labels = make_classification(n_samples = 10000 ,
                                                   n_features = 3 ,
                                                   n_redundant = 0 ,
                                                   n_classes = 5 ,
                                                   n_informative = 3 ,
                                                   n_clusters_per_class = 1 ,
                                                   class_sep = 3 ,
                                                   random_state = 10)


train_samples , test_samples , train_labels , test_labels = train_test_split(samples , labels , test_size = 0.2 , random_state = 7)
#print(train_samples.shape)#[1000 , 3]==>[n_samples , n_features]
#print(train_labels.shape)#[1000 , ]

#参数解释:
#n_samples:样本数
#n_features:特征数 = n_redundant + n_informative + n_repeated
#n_redundant:冗余信息数
#n_classes:类别数
#n_informative:有用信息数
# n_clusters_per_class:某一个类别是由多少个cluser组成

fig = plt.figure()
ax = Axes3D(fig , rect = [0 , 0 , 1 , 1] , elev = 20 , azim = 20)
ax.scatter(train_samples[: , 0] ,
           train_samples[: , 1] ,
           train_samples[: , 2] ,
           marker = 'o' ,
           c = train_labels)

plt.title("Demo Data Map")
plt.show()
