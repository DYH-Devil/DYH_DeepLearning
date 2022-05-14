from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def getdata() :
    data = load_iris()
    features = data['data']
    labels = data['target']
    train_data , test_data , train_labels , test_labels = train_test_split(features , labels , test_size = 0.2)
    return train_data , test_data , train_labels , test_labels


# train_data , test_data , train_labels , test_labels = getdata()#获取数据集
# print(train_data)
# print(train_labels)
#
# print(test_data)
# print(test_labels)