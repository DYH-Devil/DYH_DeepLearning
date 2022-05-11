#-*- coding:utf-8 -*-
import jieba
import jieba as jb
from chat_service import config

sentence = 'python和人工智能 和 c++哪个更好'
path = config.user_dict_path

jieba.load_userdict(path)
ret = jb.lcut(sentence)

print(ret)