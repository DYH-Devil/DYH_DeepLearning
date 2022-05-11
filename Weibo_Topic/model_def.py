import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel


"""
1.加载数据(已进行分词后的微博语料)
"""
file_path = r'D:\weibo_word_split.txt'
file = open(file_path , 'r' , encoding = 'utf-8' , errors = 'ignore').readlines()


"""
2.将文档转化为一个二元列表，其中一个子列表代表一条微博
"""
dataset = []#大列表
for line in file :
    weibo = []
    for word in line.split() :
        if '\u200b\u200b\u200b' in word :
            word = ''
        weibo.append(word)
        if '' in weibo :
            weibo.remove('')
    dataset.append(weibo)


"""
3.构建向量词典，语料向量化表示
"""
dictionary = corpora.Dictionary(dataset)
corpus = [dictionary.doc2bow(text) for text in dataset ]
#text表示的是大列表中的小列表子集，也就是一条微博分词后的结果
#print(corpus[0])结果表示为第i条微博中的每个单词在整个文本集中出现的次数


def coherence(num_topics) :
    ldamodel = LdaModel(corpus , num_topics = num_topics , id2word = dictionary ,  random_state = 1)
    #参数含义:corpus:词频字典:[编号:词频]，主题数量，字典映射,迭代次数
    #print(ldamodel.print_topics(num_topics = num_topics , num_words = 10))#参数:主题数，每个主题的单词数
    ldacm = CoherenceModel(model = ldamodel , texts = dataset , dictionary = dictionary , coherence = 'c_v')
    #参数含义:model:模型  texts:文本集  dictionary:字典
    return ldacm.get_coherence()


if __name__ == '__main__':
    """
    4.构建LDA模型
    """
    ldamodel = LdaModel(corpus, num_topics=10, id2word=dictionary, random_state=0)
    print(ldamodel.print_topics(num_topics=10, num_words=10))

    """
    5.将话题数量反映在图上
    """
    x = 10
    y = coherence(10)#对应的coherence值
    print(y)
    plt.scatter(x , y , c = 'r')
    plt.xlabel('话题数目')
    plt.ylabel('coherence值')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.title('coherence值与主题数量变化情况')

    plt.show()