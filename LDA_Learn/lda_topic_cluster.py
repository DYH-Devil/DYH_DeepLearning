from gensim.models import LdaModel , TfidfModel , LsiModel
from gensim import similarities
from gensim import corpora

def create_corpus(corpus_path) :#构建语料库，把读取文本内容放入列表corpus中
    texts = []
    for line in open(corpus_path , 'r' , encoding = 'utf-8').readlines() :
        word_list = []
        line = line.strip('\n')#去掉后面的换行符
        line = line.split('\t')[-1]#只保留内容，去文本序号
        line = line.strip()
        for word in line.split() :
            word_list.append(word)
        texts.append(word_list)
    return texts

def create_dict(texts) :#根据文本,创建词典
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary , corpus

corpus_train = './text_corpus/corpus_train.txt'
texts = create_corpus(corpus_train)

dictionary , corpus = create_dict(texts)

lda = LdaModel(corpus = corpus , num_topics = 10 , id2word = dictionary , passes = 30 , random_state = 1)
print(lda.print_topics(num_topics = 10 , num_words = 15))

