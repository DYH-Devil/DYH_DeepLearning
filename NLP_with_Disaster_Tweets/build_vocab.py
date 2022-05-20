#The target of this page:构建词典，统计文本词频，将每条文本转为序列

class word_Dict :
    UNK_TAG = 'UNK'#用于表示未知字符
    PAD_TAG = 'PAD'#表示填充字符

    UNK = 0#UNK_TAG对应值
    PAD = 1#PAD_TAG对应值

    count = {}#count用于统计词频

    def __init__(self):#初始化词典，最开始只放UNK和TAG
        self.dict ={
            self.UNK_TAG : self.UNK ,
            self.PAD_TAG : self.PAD
        }


    def fit(self , sentence) :#用于统计句子中的词频
        """
        :param sentence: 输入为句子
        """
        for word in sentence :
            self.count[word] = self.count.get(word , 0)  + 1#这里的意思是，从count词频表中找word对应的值，若找到则该值+1(词频+1)，若未找到(则为新词),值为0 + 1

    def build_dict(self , min = 30 , max = None , max_features = None):#实现将count中的词频映射到字典dict中
        """
        :param min: 过滤的最小词频
        :param max: #过滤的最大词频
        :param max_features: 保留的最大特征。即:只将前max_features词频存入
        """
        if min is not None :#过滤下界
            self.count = {key : value for key , value in self.count.items() if value > min}

        if max is not None :#过滤词频上界
            self.count = {key : value for key , value in self.count.items() if value < max}

        if max_features is not  None :
            self.count = dict(sorted(self.count.items() , key = lambda x : x[-1] , reverse = False)[:max_features])#对词频按递减排序，取前max_features个

        #将词频映射到词典dict
        for word in self.count.keys() :
            self.dict[word] = len(self.dict)#实际上每个词的编号就是当前dict词典的长度

        self.dict_inverse = dict(zip(self.dict.values() , self.dict.keys()))#生成反转词典，由编号对应词语

    def transform(self , sentence , max_len):#将句子转为数字序列
        """
        :param sentence: 句子序列:[word1,wor2.....]
        :param max_len: 所限制的最大句长，超出则截断，不足则补充pad
        """
        if max_len is not None :
            if max_len > len(sentence) :#句长度不足，则补充pad字符
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
            if max_len < len(sentence) :#句长超出，需截断
                sentence = sentence[:max_len]#分片操作

            return [self.dict.get(word , self.UNK) for word in sentence]#这里的意思是在dict词典中查每个word的值，找不到则标识为未知单词，返回UNK

    def transform_inverse(self , indices):#反转换，将数字序列转为单词
        return [self.dict_inverse.get(indice) for indice in indices]#从反转词典dict_inverse中查询序列值对应的单词

    def __len__(self):#求词典长度
        return len(self.dict)

