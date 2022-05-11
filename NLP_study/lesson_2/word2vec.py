import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter
from collections import Counter
import numpy as np
import random
import math
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import lib

#为了保证实验结果能够复现，我们通常将random seed设置成一个固定的数值
np.random.seed(53113)
random.seed(53113)
torch.manual_seed(53113)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA :
    torch.cuda.manual_seed(53113)


def word_tokenize(text) :#将文本转为一个一个单词，分词
    return text.split()#直接按空格分割，因为都是英文


with open(r'D:/data/text8/text8/text8.train.txt') as fin :
    text = fin.read()

text = [w for w in word_tokenize(text.lower())]#分词列表
#print(text)

vocab = dict(Counter(text).most_common(lib.MAX_VOCAB_SIZE - 1))#vocab词典中只记录text中词频最高的前MAX_VOCAB_SIZE个单词
#print(len(vocab))
#print(vocab)# 生成单词:词频 映射

vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))#其余词全部标为<unk>其词频为词频总数
#print(vocab)# <unk>:617111

idx_to_word = [word for word in vocab.keys()]#由下标-->词语
#print(idx_to_word)
word_to_idx = {word : idx for idx , word in enumerate(idx_to_word)}#由词语-->下标
#print(word_to_idx)

word_counts = np.array([count for count in vocab.values()] , dtype = np.float32)#词频
#print(word_counts)

word_freqs = word_counts / np.sum(word_counts)#每个词语的频率
#print(word_freq)

word_freqs = word_freqs ** (3./4.)#由论文得出的结论，原因不明
word_freqs = word_freqs / np.sum(word_freqs)#用来做负采样
print(word_freqs)