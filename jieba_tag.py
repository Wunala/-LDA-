# -*- coding: utf-8 -*-

import jieba
import jieba.posseg
import jieba.analyse


def read_text(file_name):
    try:
        with open(f'问询函_txt/{file_name}.txt') as txt:
            lines = txt.readlines()
        a = ''  # 空字符（中间不加空格）
        for line in lines:
            a += line.strip()  # strip()是去掉每行末尾的换行符
        c = a.split()  # 将a分割成每个字符串
        final_result = ''.join(c)  # 将c的每个字符不以任何符号直接连接
        return final_result
    except:
        pass


def cut_and_keepspecific_words(strr):
    cut = jieba.posseg.cut(sentence=strr)
    output = []
    for i in cut:
        if i.flag in ['a', 'n', 'ns', 'nt', 'nz', 'y']:
            output.append(i.word)

    result = ' '.join(output)
    return result
