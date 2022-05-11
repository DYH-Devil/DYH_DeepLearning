"""
1.分词(已实现)
2.词语存入词典，并过滤，统计词数
3.实现文本转序列
4.实现序列撰文本
"""

class word_dict :

    UNK_TAG = 'UNK'#未知字符
    PAD_TAG = 'PAD'

    UNK = 0
    PAD = 1

    count = {}#用于统计词频

    def __init__(self):#初始化
        self.dict = {
            self.UNK_TAG : self.UNK ,
            self.PAD_TAG : self.PAD
        }

    def fit(self , sentence):#统计句子词频
        for word in sentence :
            self.count[word] = self.count.get(word , 0) + 1#若没有出现过该词语，则置为0后加1，否则直接加1


    def build_dict(self , min = 5 , max = None , max_features = None):#实现将词语存入字典中
    #1.过滤词频值，特征数
        if min is not  None :
            self.count = {key : value for key , value in self.count.items() if value > min}

        if max is not None :
            self.count = {key : value for key , value in self.count.items() if value < max}

    #2.过滤最大特征数
        if max_features is not None :
            self.count = dict(sorted(self.count.items() , key = lambda x : x[-1] , reverse = False)[:max_features])#对词频进行排序，取前max_feature个

    #3.把词语放入词典
        for word in self.count :
            self.dict[word] = len(self.dict)#当前长度即该词语的编号

    #4.生成反转词典
        self.dict_inverse = dict(zip(self.dict.values() , self.dict.keys()))


    def transform(self , sentence , max_len = None) :#之前需考虑句子长度应统一(填充或裁剪)
        """
        把句子转化为序列
        :param sentence: [word1 , word2 , word3 ......]
        :param maxlen:最大句子长度
        :return:
        """
        if max_len is not  None :
            if max_len > len(sentence) :#需填充
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
            if max_len < len(sentence) :#需裁剪
                sentence = sentence[ : max_len]

        return [self.dict.get(word , self.UNK) for word in sentence]

    def transform_inverse(self , indices) :
        """
        把序列转化为句子
        :param indices: [1,2,3,4.....]
        :return:
        """
        return [self.dict_inverse.get(index) for index in indices]


    def __len__(self):
        return len(self.dict)

if __name__ == '__main__':
    ws = word_dict()
    ws.fit(['我','是','谁'])
    ws.fit(['我','是','我'])#统计词频，生成count词典

    #build_dict是基于count之上生成的，因此在这之前应该先fit生成count词频典
    ws.build_dict(min = 0)#生成词典dict
    #print(ws.dict)
    ret = ws.transform(['我','爱','北京'] , max_len = 10)
    print(ret) # 2 0 0 1 1 1 ......因为爱 北京没有在词典中出现

    opt = ws.transform_inverse(ret)
    print(opt)