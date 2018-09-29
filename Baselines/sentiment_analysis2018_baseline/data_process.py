#!/user/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import jieba
import logging
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)
jieba.load_userdict(config.user_dict_path)

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
