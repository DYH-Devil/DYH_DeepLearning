import joblib
import matplotlib.pyplot as plt
from tf_idf import tfidf_matrix
from sklearn.decomposition import PCA

kmeans = joblib.load('./model/model.pkl')

labels = []
i = 0
while i < len(kmeans.labels_) :
    labels.append(kmeans.labels_[i])
    i += 1

y_pred = kmeans.labels_

#数据降维
pca = PCA(n_components = 2)#将数据降为2维

print(tfidf_matrix.shape)

newData = pca.fit_transform(tfidf_matrix)
print(newData.shape)
