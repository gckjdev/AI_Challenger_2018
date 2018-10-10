#!/user/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import jieba
import logging
import config
import argparse
import random
import codecs
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)
jieba.load_userdict(config.user_dict_path)

PADDING_WORD_INDEX = 0
UNKNOWN_WORD_INDEX = 1

VOCAB_NUMBER = 15000    # TODO to be adjust

# 加载数据
def load_data_from_csv(file_name, header=0, encoding="utf-8", nrow=None):
    logger.info("load csv " + file_name)
    data_df = pd.read_csv(file_name, header=header, encoding=encoding, nrows=nrow)
    logger.info("load csv data info done")
    logger.info(data_df.info())
    return data_df


# 分词
def seg_words(contents):
    contents_segs = list()
    for content in contents:
        segs = jieba.lcut(content)
        contents_segs.append(" ".join(segs))

    return contents_segs


# 分词并且生词词库及其索引
def sentences_to_indices(contents):
    max_len = 0
    vocab = {}          # 词库 to 索引
    index_word = []     # 索引 to 词
    content_sequences = [] # 
    index = 2;          # after unknown word index and padding word index
    for content in contents:
        segs = jieba.lcut(content)

        # update max len
        if len(segs) > max_len:
            max_len = len(segs)

        seq = []
        for seg in segs:
            if seg not in vocab:
                index_word.append(seg)
                vocab[seg] = index
                index += 1
            seq.append(vocab[seg])
        content_sequences.append(np.array(seq))

    return max_len, index_word, vocab, content_sequences

# 根据词汇表分词并生成序列
def sentences_to_sequence(contents, vocab):
    content_sequences = [] 
    for content in contents:
        segs = jieba.lcut(content)

        seq = []
        for seg in segs:
            if seg in vocab:
                seq.append(vocab[seg])
            else:
                seq.append(UNKNOWN_WORD_INDEX)
        content_sequences.append(np.array(seq))

    return content_sequences


def get_embeding_weights(vocab, path, topn):

    # vocab={} # 词汇表为数据预处理后得到的词汇字典

    # 构建词向量索引字典
    ## 读入词向量文件，文件中的每一行的第一个变量是单词，后面的一串数字对应这个词的词向量
    # glove_dir="./data/zhwiki_2017_03.sg_50d.word2vec"
    # f=open(glove_dir,"r",encoding="utf-8")
    ## 获取词向量的维度,l表示单词数，w为某个单词转化为词向量后的维度,注意，部分预训练好的词向量的第一行并不是该词向量的维度
    # l,w=f.readline().split()

    ## 创建词向量索引字典
    # embeddings_index={}
    # print(vocab.keys()[0])

    lines_num, dim = 0, 0 # dim is word dim here, 300 dim for word2vec data here
    vectors = {} # word-vector dict
    iw = []  # index-word array
    wi = {}  # word-index dict
    logger.info("Word2Vec loading %s" % path)
    with codecs.open(path, 'r', 'utf-8') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])     # get dim from first line data
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')           
            if vocab.__contains__(tokens[0]):
                # logger.info("find vector for word %s" % tokens[0])
                vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
                # iw.append(tokens[0])    # only add words in vocab

#             if lines_num < 10:
#                 vocab[tokens[0]] = lines_num * 100               
#                 print (tokens[0], vocab)

            if topn > 0 and lines_num >= topn:
                break
            
            if lines_num % 10000 == 0:
                print("processing ", lines_num)
                if len(vocab) == len(vectors):
                    print("all words are found, break")
                    break

    # for i, w in enumerate(iw):
    #    wi[w] = i

    logger.info("Word2Vec loading.... total %d words read" % len(vectors))

    # return vectors, iw, wi, dim

    # 构建词向量矩阵，预训练的词向量中没有出现的词用0向量表示
    ## 创建一个0矩阵，这个向量矩阵的大小为（词汇表的长度+2，词向量维度）, 0索引不使用
    size = VOCAB_NUMBER       # 至少必须是词汇表的长度+2 , 2 = padding word / unknown word
    embedding_matrix = np.zeros((size, dim))
    ## 遍历词汇表中的每一项
    for word,i in vocab.items():
        ## 在词向量索引字典中查询单词word的词向量
        embedding_vector=vectors.get(word)
        ## 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
        if embedding_vector is not None:
            # logger.info("find vector for word %s" % word)
            embedding_matrix[i]=embedding_vector

    return embedding_matrix

def save_data(data, name):
    logger.info("save %s", name)
    np.save(name, np.array(data))

def load_data(name):
    logger.info("load %s", name)
    return np.load(name)