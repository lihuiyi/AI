# -*- coding: utf-8 -*-

from tencent_ai.interface_authentication import calling_api



def get_participle(data, method):
    """
    功能：获取分词的结果
    参数：
        data：调用腾讯AI开放平台的API返回的数据
        method：分词的方式，分别是："基础词"、"混排词"
    返回值：分词后的结果(str类型)
    """
    tokens = None
    if method == "基础词":
        tokens = data["base_tokens"]
    elif method == "混排词":
        tokens = data["mix_tokens"]
    all_words = []
    for word_dict in tokens:
        word = word_dict["word"]
        all_words.append(word)
    all_words = "     ".join(all_words)
    return all_words



#
# # 分词的API地址
# url = "https://api.ai.qq.com/fcgi-bin/nlp/nlp_wordseg"
# encode = "gbk"
# app_id = '2108919620'
# app_key = 'JMmBTRnHkFi7hh8q'
# text = "腾讯人工智能"
# text = text.encode(encode)
# params = {
#     "text": text
# }
# # 分词的方式
# method = "混排词"  # 可选值："基础词"、"混排词"
# # 调用腾讯AI开放平台的API
# data = calling_api(app_id, app_key, params, url, encode)
# # 获得分词的结果
# all_words = get_participle(data, method)
# print(all_words)


import jieba
text = "这个把手该换了，我不喜欢日本和服，别把手放在我的肩膀上，工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
a = "  ".join(jieba.cut(text))
print(a)




