"""
定义dataset类
"""
import torch
import torch.utils.data as tud
import lib
import word2vec
from torch.utils.data import DataLoader

class WordEmbeddingDataset(tud.Dataset) :
    def __init__(self , text , word_to_idx , idx_to_word , word_freqs , word_counts) :
        """
        :param text: 单词表
        :param word_to_idx: 单词 to 索引 映射
        :param idx_to_word: 索引 to 单词 映射
        :param word_freqs: 词频率
        :param word_counts: 词数统计
        """
        super(WordEmbeddingDataset , self).__init__()
        self.text_encoded = [word_to_idx.get( t , lib.MAX_VOCAB_SIZE - 1) for t in text]#对单词进行编码，其对应的码就是其下标或max-1
        self.text_encoded = torch.LongTensor(self.text_encoded)#转为LongTensor
        self.word_to_idx = word_to_idx
        self.word_to_idx = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self) :
        #返回dataSet长度
        return len(self.text_encoded)

    def __getitem__(self, idx) :
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx - lib.C , idx)) + list(range(idx + 1 , idx + lib.C + 1))#周围词位置
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]#防止位置溢出
        pos_words = self.text_encoded[pos_indices]#中心词(正采样)
        neg_words = torch.multinomial(self.word_freqs , lib.K * pos_words.shape[0] , True)#负采样
        #输入:词频  样本数:这里设置为K * 正样本个数:意思是每采集一个正样本，就需要K个负样本 , inplace = True
        return center_word , pos_words , neg_words#返回的一条dataset里包含(中心词，周围词，负样本)


#实例化dataset
dataset = WordEmbeddingDataset(word2vec.text , word2vec.word_to_idx , word2vec.idx_to_word , word2vec.word_freqs , word2vec.word_counts)

#实例化dataloader
dataloader = DataLoader(dataset = dataset , shuffle = True , batch_size = lib.BATCH_SIZE)
for idx , (center_word , pos_words , neg_words) in enumerate(dataloader) :
    print(center_word , pos_words , neg_words)
    break