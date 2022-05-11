import torch
import torch.nn as nn
class EmbeddingModel(nn.Module) :
    def __init__(self , vocab_size , embed_size) :
        """
        :param vocab_size: 词典中词语数目
        :param embed_size: 维度，用多少维表示一个单词
        """
        super(EmbeddingModel , self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size , self.embed_size , sparse = False)
        self.out_embed.weight.data.uniform(-initrange , initrange)#随机生成权重

        self.in_embed = nn.Embedding(self.vocab_size , self.embed_size , sparse = False)
        self.in_embed.weight.data.uniform(-initrange , initrange)


    def forward(self , input_labels , pos_labels , neg_labels):
        """
        :param input_labels: 中心词[batchsize]
        :param pos_labels: 中心词周围窗口范围内出现过的单词[batchsize * (C * 2)]
        :param neg_labels: 中心词周围窗口范围内未出现过的单词[batchsize * (C * K)](负样本)
        :return:loss , [batchsize]
        """
        batch_size = input_labels.size(0)
        input_embedding = self.in_embed(input_labels)
        pos_embedding = self.out_embed(pos_labels)
        neg_labels = self.out_embed(neg_labels)

