import gensim
import stop_words
import nltk
from nltk.stem.porter import PorterStemmer
from gensim import corpora , models

doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."


doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]# 将文档组合为列表
word_list = []#词库

"""
一.清洗文档
1.分词
2.停用词
3.词干抽取
"""

#1.分词
tokenizer = nltk.RegexpTokenizer(r'\w+')#TODO:nltk
for doc in doc_set :
    doc_lower = doc.lower()
    word_set = tokenizer.tokenize(doc_lower)
    word_list.append(word_set)


print("word_list:" , word_list)


#2.停用词
stop_words = stop_words.get_stop_words('english')#参数传入的是一种语言的代码，如english
word_list = [i for i in word_list if i not in stop_words]#停用词剔除
print("word_list_stop:" , word_list)


#3.词干提取
"""
词干提取会将单词去除词缀而得到词根，因为如果一个词汇以多种形式存在于句子中，对主题模型会产的生干扰，(当然只是在英语中)
nltk包就是实现这一目的
"""
texts = []
p_stem = PorterStemmer()
for word_set in word_list :
    word_stem = [p_stem.stem(i) for i in word_set]
    texts.append(word_stem)

print("word_list_stem:" , texts)


"""
二.构建文档词频矩阵document-term matrix
（译注：document-term matrix 是一个描述文档词频的矩阵，每一行对应文档集中的一篇文档，每一列对应一个单词，这个矩阵可以根据实际情况，采用不同的统计方法来构建。）
"""
dictionary = corpora.Dictionary(texts)#Dictionary参数为文档集，规模就是[[w1,w2.....],[v1,v2......]]大列表表示文档集，大列表中的小列表表示一篇文档的词集
print(dictionary.token2id)#查看每个词的id



"""
三.构建词袋
doc2bow() 方法将 dictionary 转化为一个词袋。得到的结果 corpus 是一个向量的列表，向量的个数就是文档数。在每个文档向量中都包含一系列元组。
"""
doc2bow = [dictionary.doc2bow(text) for text in texts]
print("文档集的词袋" , doc2bow)
#第一个元组(0,2)中表示的的即:id为0的单词brocolli的词频为2

LDA_Model = models.LdaModel(corpus = doc2bow , num_topics = 3 , id2word = dictionary , passes = 20)
"""
corpus:文档即的词袋
num_topics:目标种类数目
id2word:单词:idx的映射
passes:遍历次数，越多次越精确，但时间花销越大
"""
print(LDA_Model.print_topics(num_topics = 10 , num_words = 10))