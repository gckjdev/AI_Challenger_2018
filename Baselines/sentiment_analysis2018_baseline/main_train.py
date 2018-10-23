#!/user/bin/env python
# -*- coding:utf-8 -*-

from data_process import load_data_from_csv, seg_words, get_embeding_weights, sentences_to_indices, save_data, load_data
from data_process import convert_label_to_index, convert_index_to_label
from model import TextClassifier, build_rnn_model, trainRNNModel, predictRNNModel, load_rnn_model
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

    parser.add_argument('-lm', '--load_model', type=int, nargs='?',
                        help='load model or not')

    parser.add_argument('-t', '--test', type=int, nargs='?',
                        help='test mode or not')

    parser.add_argument('-e', '--epochs', type=int, nargs='?',
                        help='train epochs')                       

    args = parser.parse_args()
    model_name = args.model_name
    if not model_name:
        model_name = "model_dict.pkl"

    load_cache = True
    if args.load_cache is None:
        load_cache = True
    else:
        load_cache = False if args.load_cache == 0 else True

    is_load_model = True
    if args.load_model is None:
        is_load_model = True
    else:
        is_load_model = False if args.load_model == 0 else True

    is_test = True
    if args.test is None:
        is_test = True
    else:
        is_test = False if args.test == 0 else True

    epochs = args.epochs
    if epochs is None:
        epochs = 3

    logger.info("test mode is %s" % is_test)
    logger.info("load cache is %s" % load_cache)
    logger.info("train epochs is %s" % epochs)

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

    # load train data and validate data
    logger.info("start load data")
    traing_num = 10000 if is_test else None
    validate_num = 5000 if is_test else None        
    train_data_df = load_data_from_csv(config.train_data_path, nrow=traing_num)
    validate_data_df = load_data_from_csv(config.validate_data_path, nrow=validate_num)

    # get all train sentences
    content_train = train_data_df.iloc[:, 1]
    logger.info(content_train[0])
    logger.info(content_train[1])

    logger.info("start seg train sentences to vector")
    if not load_cache:
        max_len, word, vocab, sequences = sentences_to_indices(content_train)
        save_data(vocab, "all_vocab.npy")
        save_data(word, "word.npy")
        # save_data(sequences, "seq.npy")

    word = load_data("word.npy").tolist()
    vocab = load_data("all_vocab.npy").tolist()
    # sequences = load_data("seq.npy").tolist()

    logger.info("all vocab len %d" % len(vocab))
    logger.info("word count %d" % len(word))
    # logger.info("sequence len %d" % len(sequences))

    if not load_cache:
        embedding_matrix, vocab = get_embeding_weights(vocab, config.word2vec_path, 0)
        save_data(embedding_matrix, "emb.npy")
        save_data(vocab, "train_vocab.npy")
    embedding_matrix = load_data("emb.npy")
    vocab = load_data("train_vocab.npy").tolist()
    logger.info("train vocab len %s" % len(vocab))
    data_process.set_vocab_number(len(vocab))
    print(embedding_matrix[0])
    print(embedding_matrix[1])    

    logger.info("start seg train data")
    # segment sentences to words
    # content_train = seg_words(content_train)
    content_train = train_data_df.iloc[:, 1]
    print(content_train[0])
    logger.info("total %d train data" % len(content_train))
    if not load_cache:
        sequences = data_process.sentences_to_sequence(content_train, vocab)
        save_data(sequences, "seq.npy")
    sequences = load_data("seq.npy").tolist()
    logger.info("complete seg train data")    

    # get column names
    columns = train_data_df.columns.values.tolist()
    # logger.info(columns)

    

    # use RNN to train and predict
    rnn_model_dict = dict()
    for column in columns[2:]:   # 逐列遍历每一个训练的标注 label
        
        label_train = np_utils.to_categorical(convert_label_to_index(train_data_df[column]), num_classes=data_process.NUM_CLASS)
        logger.info(label_train[:10])
        logger.info(train_data_df[column][:10])
        # logger.info(label_train[1])

        content_train = sequences

        model = build_rnn_model(data_process.VOCAB_NUMBER, embedding_matrix, data_process.NUM_CLASS)

        weights_name = column + ".h5"
        if is_load_model:
            load_rnn_model(model, weights_name)
        else:
            trainRNNModel(model, content_train, label_train, weights_name, epochs)

        rnn_model_dict[column] = model
        if is_test:
            break

    # use RNN model to validate
    content_validate = validate_data_df.iloc[:, 1]

    logger.info("load RNN validate data sentences...")
    content_validate = data_process.sentences_to_sequence(content_validate, vocab)
    print(validate_data_df.iloc[:, 1][0])
    print(validate_data_df.iloc[:, 1][1])
    print(content_validate[0])
    print(content_validate[1])


    for column in columns[2:]:
        logger.info("start RNN validate model for %s" % column)
        label_validate = np_utils.to_categorical(convert_label_to_index(validate_data_df[column]), num_classes = data_process.NUM_CLASS)
        logger.info(label_validate[:10])
        logger.info(validate_data_df[column][:10])
        # logger.info(label_validate[1])
        score = predictRNNModel(rnn_model_dict[column], content_validate, label_validate)
        if is_test:
            break

    # logger.info("start train feature extraction")
    # vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=5, norm='l2')
    # vectorizer_tfidf.fit(content_train)
    # logger.info("complete train feature extraction models")
    # logger.info("vocab shape: %s" % np.shape(vectorizer_tfidf.vocabulary_.keys()))




    # # model train
    # logger.info("start train model")
    # classifier_dict = dict()
    # for column in columns[2:]:   # 逐列遍历每一个训练的标注 label
    #     label_train = train_data_df[column]
    #     logger.info("content train first %s" % content_train[0])
    #     logger.info("label train first %s" % label_train[0])
    #     text_classifier = TextClassifier(vectorizer=vectorizer_tfidf)
    #     logger.info("start train %s model" % column)
    #     text_classifier.fit(content_train, label_train)
    #     logger.info("complete train %s model" % column)
    #     classifier_dict[column] = text_classifier
    #     if is_test:
    #         logger.info("test, only run once")
    #         break

    logger.info("complete train model")

    # # validate model
    # content_validate = validate_data_df.iloc[:, 1]

    # logger.info("start seg validate data")
    # content_validate = seg_words(content_validate)
    # logger.info("complete seg validate data")

    # logger.info("start validate model")
    # f1_score_dict = dict()
    # for column in columns[2:]:
    #     label_validate = validate_data_df[column]

    #     # predict and save f1 score
    #     text_classifier = classifier_dict[column]
    #     f1_score = text_classifier.get_f1_score(content_validate, label_validate)

    #     f1_score_dict[column] = f1_score

    # # get overall f1 score
    # f1_score = np.mean(list(f1_score_dict.values()))
    # str_score = "\n"
    # for column in columns[2:]:
    #     str_score = str_score + column + ":" + str(f1_score_dict[column]) + "\n"

    # logger.info("f1_scores: %s\n" % str_score)
    # logger.info("f1_score: %s" % f1_score)
    # logger.info("complete validate model")

    # # save model
    # logger.info("start save model")
    # model_save_path = config.model_save_path
    # if not os.path.exists(model_save_path):
    #     os.makedirs(model_save_path)

    # joblib.dump(classifier_dict, model_save_path + model_name)
    # logger.info("complete save model")


