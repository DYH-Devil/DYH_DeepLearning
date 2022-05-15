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


class LDA_topic() :
    def __init__(self , corpus , dictionary , num_topics , num_words):
        self.corpus = corpus
        self.dictionary = dictionary
        self.num_topics = num_topics
        self.num_words = num_words
        self.lda = LdaModel(corpus = self.corpus , id2word = self.dictionary , num_topics = num_topics , passes = 30 , random_state = 1)

    def cluster(self):
        key_word_file = './data/keywords.txt'
        save_file = open(key_word_file , 'w' , encoding = 'utf-8')#保存路径
        cluster_res = self.lda
        for topic in cluster_res.print_topics(num_topics = self.num_topics , num_words = self.num_words) :
            words = []
            for word in topic[1].split('+') :
                word = word.split('*')[1].replace(' ', '')
                words.append(word)
            save_file.write(str(topic[0]) + '\t' + ','.join(words) + '\n')

if __name__ == '__main__':
    corpus_train = './text_corpus/corpus_train.txt'
    texts = create_corpus(corpus_train)

    dictionary , corpus = create_dict(texts)#corpus:[词语编号id,文本中出现次数count]
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda = LDA_topic(corpus_tfidf , dictionary , 10 , 30)
    lda.cluster()



