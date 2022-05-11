import jieba
from chat_service import config
import string
import jieba.posseg as psg

path = config.user_dict_path

letters = string.ascii_lowercase + '+'

jieba.load_userdict(path)#加载词典

def cut_sentence_by_word(sentence) :#按字分词
    """
    中文按字分词
    英文按单词分词
    :param sentence: 句子
    :return:
    """
    result = []
    temp = ""
    for word in sentence :
        if word.lower() in letters :
            temp += word#英文字符，连接

        else :
            if temp != '' :#出现非英文
                result.append(temp.lower())
                temp = ""#temp置空

            #若temp为空则说明这个字符是中文
            result.append(word.strip())

    if temp != "" :#判断最后字符串是否为英文单词
        result.append(temp.lower())
    return  result




def cut_sentence(sentence , by_word = False , use_stopword = False , with_psg = False) :#分词
    """
    #实现中英文分词
    :param sentence:句子
    :param use_stopword:使用停词
    :param with_sg:词性
    :return:切分结果
    """
    if by_word :
        result = cut_sentence_by_word(sentence)

    else :
        if with_psg :
            result = psg.lcut(sentence)#需要词性
        else:
            result = jieba.lcut(sentence)#不需要词性

    if use_stopword :
        result = [i for i in result if i not in  stop_words]
    return  result

if __name__ == '__main__':
    sentence = "python和c++哪个难?haha"
    print(cut_sentence(sentence ,with_psg = True))

