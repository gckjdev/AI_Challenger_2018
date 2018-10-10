#!/user/bin/env python
# -*- coding:utf-8 -*-

from data_process import load_data_from_csv, seg_words, get_embeding_weights, sentences_to_indices, save_data, load_data
from model import TextClassifier, buildRNNModel, trainRNNModel, predictRNNModel
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
import data_process
from keras.utils import np_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, nargs='?',
                        help='the name of model')

    parser.add_argument('-lc', '--load_cache', type=int, nargs='?',
                        help='load cache or not')

    parser.add_argument('-t', '--test', type=int, nargs='?',
                        help='test mode or not')

    args = parser.parse_args()
    model_name = args.model_name
    if not model_name:
        model_name = "model_dict.pkl"

    load_cache = True
    if args.load_cache is None:
        load_cache = True
    else:
        load_cache = False if args.load_cache == 0 else True

    is_test = True
    if args.test is None:
        is_test = True
    else:
        is_test = False if args.test == 0 else True

    logger.info("test mode is %s" % is_test)
    logger.info("load cache is %s" % load_cache)

#    read_word2vec.read_vectors(config.word2vec_path, 10000)  # total 1292679

    # contents = [u"人生就是这样子", u"人不可无志气", u"不可不要", u"美好的生活", u"人无完人", u"大众点评"]
    # m, w, v, s = sentences_to_indices(contents)
    # print("max len ", m)
    # print(w, v)
    # print(s)

    # a = np.array(s)
    # print("save ", a)
    # np.save("a.npy", a)
    # b = np.load("a.npy")
    # print("load ", b)

    # vocab = { u"你好" : 0, u"朋友" : 1, u"人" : 2 , u"年":3, u"一个":4}
    # embedding_matrix = get_embeding_weights(vocab, config.word2vec_path, 1000000)
    # print(vocab)
    # print(embedding_matrix)

    # load train data
    logger.info("start load data")
    traing_num = 1000 if is_test else None
    validate_num = 100 if is_test else None
    train_data_df = load_data_from_csv(config.train_data_path, nrow=traing_num)
    validate_data_df = load_data_from_csv(config.validate_data_path, nrow=validate_num)

    # get all train sentences
    content_train = train_data_df.iloc[:, 1]

    logger.info("start seg sentences to vector")
    if not load_cache:
        max_len, word, vocab, sequences = sentences_to_indices(content_train)
        save_data(vocab, "vocab.npy")
        save_data(word, "word.npy")
        save_data(sequences, "seq.npy")

    max_len = 100000 # TODO to be changed
    word = load_data("word.npy").tolist()
    vocab = load_data("vocab.npy").tolist()
    sequences = load_data("seq.npy").tolist()

    logger.info("vocab len %d" % len(vocab))
    logger.info("word count %d" % len(word))
    logger.info("max len %d" % max_len)
    logger.info("sequence len %d" % len(sequences))

    if not load_cache:
        embedding_matrix = get_embeding_weights(vocab, config.word2vec_path, 0)
        save_data(embedding_matrix, "emb.npy")
    embedding_matrix = load_data("emb.npy")
    print(embedding_matrix[0])
    print(embedding_matrix[1])    

    logger.info("start seg train data")
    # segment sentences to words
    content_train = seg_words(content_train)
    logger.info("complete seg train data")    

    # get column names
    columns = train_data_df.columns.values.tolist()
    # logger.info(columns)

    NUM_CLASS = 3

    # use RNN to train and predict
    rnn_model_dict = dict()
    for column in columns[2:]:   # 逐列遍历每一个训练的标注 label
        
        label_train = np_utils.to_categorical(train_data_df[column], num_classes=NUM_CLASS)
        content_train = sequences

        model = buildRNNModel(data_process.VOCAB_NUMBER, embedding_matrix)

        name = column + ".h5"
        model = trainRNNModel(model, content_train, label_train, name)
        rnn_model_dict[column] = model

        if is_test:
            break

    logger.info("start train feature extraction")
    vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=5, norm='l2')
    vectorizer_tfidf.fit(content_train)
    logger.info("complete train feature extraction models")
    logger.info("vocab shape: %s" % np.shape(vectorizer_tfidf.vocabulary_.keys()))


    # use RNN model to validate
    content_validate = validate_data_df.iloc[:, 1]

    logger.info("start RNN validate data")
    content_validate = data_process.sentences_to_sequence(content_validate, vocab)
    print(content_validate[0])
    print(content_validate[1])

    logger.info("start RNN validate model")
    for column in columns[2:]:
        label_validate = np_utils.to_categorical(validate_data_df[column], num_classes = NUM_CLASS)
        score = predictRNNModel(rnn_model_dict[column], content_validate, label_validate)
        if is_test:
            break

    # model train
    logger.info("start train model")
    classifier_dict = dict()
    for column in columns[2:]:   # 逐列遍历每一个训练的标注 label
        label_train = train_data_df[column]
        logger.info("content train first %s" % content_train[0])
        logger.info("label train first %s" % label_train[0])
        text_classifier = TextClassifier(vectorizer=vectorizer_tfidf)
        logger.info("start train %s model" % column)
        text_classifier.fit(content_train, label_train)
        logger.info("complete train %s model" % column)
        classifier_dict[column] = text_classifier
        if is_test:
            logger.info("test, only run once")
            break

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


