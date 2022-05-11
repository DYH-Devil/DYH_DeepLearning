import numpy as np

class Num_Sequence :
    PAD_TAG = 'PAD'
    UNK_TAG = 'UNK'
    SOS_TAG = 'SOS'  # 句子开始标记
    EOS_TAG = 'EOS'  # 句子结束标记

    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {self.PAD_TAG : self.PAD ,
                     self.UNK_TAG : self.UNK ,
                     self.SOS_TAG : self.SOS ,
                     self.EOS_TAG : self.EOS}
        for i in range(10) :
            self.dict[str(i)] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values() , self.dict.keys()))

    def transform(self , sentence , max_len , addEoS = False):
        """把sentence转化为序列"""
        if len(sentence) > max_len :
            sentence = sentence[:max_len]
        sentence_len = len(sentence)#提前记录句子长度,同一个batch下句子判断前后长度一致
        if addEoS :
            sentence = sentence + [self.EOS_TAG]
        if sentence_len < max_len :
            sentence = sentence + [self.PAD_TAG] * (max_len - sentence_len)
        result = [self.dict.get(i, self.UNK) for i in sentence]
        return result

    def inverse_transform(self , indices):
        result = [self.inverse_dict.get(i , self.UNK_TAG) for i in indices]
        return  result

    def __len__(self):
        return len(self.dict)
