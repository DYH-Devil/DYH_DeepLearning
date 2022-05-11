"""
用jieba实现中文分词
"""

import jieba
import os
import re



jieba.setLogLevel(jieba.logging.INFO)


def stopwords_list() :#停用词加载
    stopwords_list = [line.strip() for line in open(r'D:/stopword/baidu_stopwords.txt' , 'r' , encoding = 'utf-8').readlines()]
    return stopwords_list


"""
数据清洗:对一些标识进行过滤的操作
"""
def processing(text) :
    filters = ['\t', '\n', '\x97', '\x96', '@', '#', '$', '%', '&','!' , '，' , '？' , '——' , '_' , '。']
    text = re.sub('|'.join(filters), ' ', text)  # |表示综合所有的filters,全部替换
    text = re.sub("@.+?( |$)", "", text)#去除@xxx用户名
    text = re.sub("【.+?】", "", text)#去除【】里的内容，通常里面的内容是引用
    text = re.sub("#.*#", "", text)  # 去除话题引用
    text = re.sub("<.*?>", "", text)
    text = re.sub("\n","",text)
    text = re.sub('[0-9]{16}' , '' , text)
    text = text.replace('*' , '')
    return text


"""
对句子实现中文分词
"""
def word_split(text):
    sentence_depart = jieba.cut(text.strip())
    stop_words = stopwords_list()#停用词
    output = ""#用于输出
    for word in sentence_depart :
        if word not in stop_words :#去停用词
            if word != '\n':
                output += word
                output += ""
    return output.strip()



file_path = r'D:\weibo_data'
out_put_file = r'D:\weibo_word_split.txt'
output = open(out_put_file , 'w' , encoding = 'utf-8')

file_list = []
for file in os.listdir(file_path) :
    file_name = os.path.join(file_path , file)
    file_list.append(file_name)

for line in open(file_list[0] , 'r' , encoding = 'utf-8').readlines() :
    line = processing(line)
    line_split = word_split(line)
    output.write(line_split + '\n')

output.close()