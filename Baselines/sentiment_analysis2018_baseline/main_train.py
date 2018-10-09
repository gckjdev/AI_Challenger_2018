#!/user/bin/env python
# -*- coding:utf-8 -*-

from data_process import load_data_from_csv, seg_words, get_embeding_weights
from model import TextClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import config
import logging
import numpy as np
from sklearn.externals import joblib
import os
import argparse
import jieba
import read_word2vec
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, nargs='?',
                        help='the name of model')

    args = parser.parse_args()
    model_name = args.model_name
    if not model_name:
        model_name = "model_dict.pkl"

#    read_word2vec.read_vectors(config.word2vec_path, 10000)  # total 1292679

    vocab = { u"你好" : 0, u"朋友" : 1, u"人" : 2 , u"年":3, u"一个":4}
    embedding_matrix = get_embeding_weights(vocab, config.word2vec_path, 1000000)
    print(vocab)
    print(embedding_matrix)

    # load train data
    logger.info("start load data")
    traing_num = 1000
    validate_num = 100
    validate_num = 100
    train_data_df = load_data_from_csv(config.train_data_path, nrow=traing_num)
    validate_data_df = load_data_from_csv(config.validate_data_path, nrow=validate_num)

    # get all train sentences
    content_train = train_data_df.iloc[:, 1]

    logger.info("start seg train data")
    # segment sentences to words
    content_train = seg_words(content_train)
    logger.info("complete seg train data")

    # get column names
    columns = train_data_df.columns.values.tolist()
    logger.info(columns)

    logger.info("start train feature extraction")
    vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=5, norm='l2')
    vectorizer_tfidf.fit(content_train)
    logger.info("complete train feature extraction models")
    logger.info("vocab shape: %s" % np.shape(vectorizer_tfidf.vocabulary_.keys()))

    # model train
    logger.info("start train model")
    classifier_dict = dict()
    for column in columns[2:]:
        label_train = train_data_df[column]
        logger.info("content train first %s" % content_train[0])
        logger.info("label train first %s" % label_train[0])
        text_classifier = TextClassifier(vectorizer=vectorizer_tfidf)
        logger.info("start train %s model" % column)
        text_classifier.fit(content_train, label_train)
        logger.info("complete train %s model" % column)
        classifier_dict[column] = text_classifier

    logger.info("complete train model")

    # validate model
    content_validate = validate_data_df.iloc[:, 1]

    logger.info("start seg validate data")
    content_validate = seg_words(content_validate)
    logger.info("complete seg validate data")

    logger.info("start validate model")
    f1_score_dict = dict()
    for column in columns[2:]:
        label_validate = validate_data_df[column]

        # predict and save f1 score
        text_classifier = classifier_dict[column]
        f1_score = text_classifier.get_f1_score(content_validate, label_validate)

        f1_score_dict[column] = f1_score

    # get overall f1 score
    f1_score = np.mean(list(f1_score_dict.values()))
    str_score = "\n"
    for column in columns[2:]:
        str_score = str_score + column + ":" + str(f1_score_dict[column]) + "\n"

    logger.info("f1_scores: %s\n" % str_score)
    logger.info("f1_score: %s" % f1_score)
    logger.info("complete validate model")

    # save model
    logger.info("start save model")
    model_save_path = config.model_save_path
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    joblib.dump(classifier_dict, model_save_path + model_name)
    logger.info("complete save model")


