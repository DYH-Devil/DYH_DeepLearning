from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def tf_idf(corpus) :
    #将文本中的词语转化为词频矩阵，矩阵元素a[i][j]表示j词在i类文本下的词频
    vectorizer = CountVectorizer(max_features = 10000)#先实例化Countervectorizer类
    #max_len ，若不为None:则只以文档中词频前max_len的词语构建词典

    #1.统计词频
    vectorizer_res = vectorizer.fit_transform(corpus)#将语料corpus转为词频矩阵

    #2.计算每个词语的tf_idf  tf_idf = tf * idf
    tf_idf_transformer = TfidfTransformer()
    tfidf = tf_idf_transformer.fit_transform(vectorizer_res)#计算tf_idf

    tfidf_matrix = tfidf.toarray()
    #print(tfidf_matrix.shape)#[len(corpus) , max_features]
    #word = vectorizer.get_feature_names()

    return tfidf_matrix

corpus = []
text = open('D:/weibo_word_split.txt', 'r', encoding='utf-8')
for line in text.read().split('\n'):#1 line 表示一条文档
    corpus.append(line)  # 把分词结果装入语料库

tfidf_matrix = tf_idf(corpus)#计算得到词频矩阵
