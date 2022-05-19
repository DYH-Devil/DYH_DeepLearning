"""
The target of this page:清洗文本数据，去除不必要的符号，表情等
"""
import re
from get_data import text_train , text_test
from torchtext.data.utils import get_tokenizer#分词器
from string import punctuation as punctuation_string

#----------------------------------------------------------------------------------------------
#step1:去除表情符号
def clean_emoji(text) :
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'EMOJI', text)

# #test:
# text = 'hellow I name is 😀'
# text = clean_emoji(text)
# res:hellow I name is EMOJI
# print(text)
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
#step2:去除引用符@以及后面的用户名
def clean_at(text) :
    at = re.compile(r'@\S+')
    return at.sub(r'username', text)#将@后的用户名用username替换

# #test
# text = 'please @DYHBadBoy thank U!'
# print(clean_at(text))
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
#step3:去除HTML标记内容
def clean_HTML(text) :
    html = re.compile(r'<.*?>')
    return html.sub(r'' , text)

# #test
# text = """<div>
# <h1>Real or Fake</h1>
# <p>Kaggle </p>
# <a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>
# </div>"""
# print(clean_HTML(text))
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
#step4:去除url链接
def cleam_URL(text) :
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'URL', text)

# #test
# text = 'Coucou, venez sur https://www.youtube.com/'
# print(cleam_URL(text))
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
#step5:去除重复的标点符号
def clean_repeat_punct(text) :
    rep = re.compile(r'([!?.,]){2,}')
    return rep.sub(r'\1 REPEAT', text)

# #test
# print(clean_repeat_punct(",,,,,,,"))
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
#step6:去除单词后拖长的字符，如:yessssss,okkkkkk
def clean_words_elong(text) :
    rep = re.compile(r'\b(\S*?)([a-z])\2{2,}\b')
    return rep.sub(r'\1\2 ELONG', text)

# #test
# text = 'yesssss okkkkkk'
# print(clean_words_elong(text))
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
#step7:去除数字
def clean_number(text):
    rep = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return rep.sub('', text)

# #test
# print(clean_number("13.5"))
#----------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
#step7:去除标点
def clean_punctuation(text):
    rep = re.sub('[{}]'.format(punctuation_string),"",text)
    return rep

# #test
# print(clean_number("13.5"))
#----------------------------------------------------------------------------------------------

#END

def text_process(text_list) :
    text_process = []
    for text in text_list :
        text = text.lower()#转小写
        text = clean_emoji(text)
        text = clean_at(text)
        text = clean_HTML(text)
        text = cleam_URL(text)
        text = clean_repeat_punct(text)
        text = clean_words_elong(text)
        text = clean_number(text)
        text = clean_punctuation(text)
        text_process.append(text)
    return text_process

def split_word(text) :#文本分词
    tokenizer = get_tokenizer('basic_english')
    word_split = []
    for line in text :
        word_split.append(tokenizer(line))#对每条推文进行分词
    return word_split

