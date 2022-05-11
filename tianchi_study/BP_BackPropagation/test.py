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


#step3:模型测试
from data_make import test_samples , test_labels#测试集
from data_make import train_samples , train_labels
BP_model = pickle.load(open('./model/model.dat' , 'rb'))#读取模型
# train_predict = BP_model.predict(train_samples)
# print(classification_report(train_labels , train_predict))

test_predict = BP_model.predict(test_samples)
print(classification_report(test_labels , test_predict))