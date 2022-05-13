from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tf_idf import tfidf_matrix
import joblib
import os

def KMeans_Topics(class_num) :#用KMeans完成聚类
    """
    :param class_num: 类数目
    :return:
    """
    kmeans = KMeans(n_clusters = class_num)#实例化
    kmeans.fit(tfidf_matrix)
    joblib.dump(kmeans, './model/model.pkl')
    return kmeans

kmeans = KMeans_Topics(5)
labels = kmeans.labels_

