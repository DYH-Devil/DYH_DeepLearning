import joblib
from tf_idf import tfidf_matrix
from sklearn.decomposition import PCA

def data_pca(tfidf_matrix) :
    #数据降维
    pca = PCA(n_components = 2)#将数据降为2维
    newData = pca.fit_transform(tfidf_matrix)
    #降维前:tfidf_matrix shape:[len(corpus) , 1000(n_features)]
    return newData
    #print(newData.shape)
    #降维后:newData.shape:[len(corpus) , 2]

newData = data_pca(tfidf_matrix)
joblib.dump(newData, './model/pca.pkl')
