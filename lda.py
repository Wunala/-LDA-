# -*- coding: utf-8 -*-
import re
import os
import jieba
import jieba.posseg
import jieba.analyse
import numpy as np
import pandas as pd
from jieba_tag import read_text
from jieba_tag import cut_and_keepspecific_words

my_dict_list = ['财经金融词汇.txt', '经济金融类词汇.txt', '490000.txt', '490100.txt', '490200.txt', '490300.txt']
for dict_name in my_dict_list:
    jieba.load_userdict('补充词典/{0}'.format(dict_name))



sh = pd.read_csv('sh.csv', encoding='utf-8')

# 获取函件的名字以便进行遍历读取txt文件
file_names = []
for i in range(sh.shape[0]):
    file_names.append(sh['函件编码'][i])


# 读取每一个txt文件并且进行读取和保留特定的词
texts = []
for file_name in file_names:
    try:
        with open(f'问询函_txt/{file_name}.txt') as txt:
            lines = txt.readlines()[7:-6]
        a = ''  # 空字符（中间不加空格）
        for line in lines:
            a += line.strip()  # strip()是去掉每行末尾的换行符
        c = a.split()  # 将a分割成每个字符串
        final_result = ''.join(c)
        cut = jieba.posseg.cut(sentence=final_result)
        output = []
        for i in cut:
            if i.flag in ['n', 'nt', 'nz', 'z']:
                output.append(i.word)

        result = ' '.join(output)
        tfidf = jieba.analyse.extract_tags
        keywards = tfidf(result)
        texts.append(' '.join(keywards))
    except:
        pass


# 对已经经过预处理的文本进行分词，并存入list
results = []
for text in texts:
    text_cut = jieba.cut(text)
    results.append(' '.join(text_cut))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# 获取词频向量
corpus = results
cntVector = CountVectorizer()
cntTf = cntVector.fit_transform(corpus)
# 输出选取词特征
vocs = cntVector.get_feature_names()
# LDA主题模型
lda = LatentDirichletAllocation(n_components=5,  # 主题个数K
                                max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
docres = lda.fit_transform(cntTf)
# 文档所属各个主题的概率
LDA_corpus = np.array(docres)
# print('类别所属概率：\n', LDA_corpus)

# 构建一个零矩阵
LDA_corpus_one = np.zeros([LDA_corpus.shape[0]])
# 对比所属两个概率的大小，确定属于的类别
LDA_corpus_one = np.argmax(LDA_corpus, axis=1) # 取概率最大的作为主题

topic_result = pd.DataFrame(columns=['函件编码', '主题'])
for i in range(len(file_names)):
    topic_result.loc[i] = [file_names[i], LDA_corpus_one[i]]
topic_result.to_csv('topic_result.csv')

# print('每个文档所属类别：', LDA_corpus_one)

# 打印每个主题中主要单词的权重值
tt_matrix = lda.components_
topic = pd.DataFrame(columns=['topic', 'top20_words'])

id = 0
for tt_m in tt_matrix:
    tt_dict = [(name, tt) for name, tt in zip(vocs, tt_m)]
    tt_dict = sorted(tt_dict, key=lambda x: x[1], reverse=True)
    # 打印权重值大于0.6的主题词：
    # tt_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0.6]
    # 打印每个类别前5个主题词：
    tt_dict = tt_dict[:20]
    top20words = pd.DataFrame(columns=['words', 'rate'])
    tem_list = []
    for i in range(20):
        top20words.loc[i] = [tt_dict[i][0], tt_dict[i][1]]
        tem_list.append(tt_dict[i][0])
    top20words.to_csv(f'topic{id}_top20words.csv')
    topic.loc[id] = [id, ' '.join(tem_list)]


    # print('主题%d:' % (id), tt_dict)
    id += 1
topic.to_csv('topic_top20words.csv')
