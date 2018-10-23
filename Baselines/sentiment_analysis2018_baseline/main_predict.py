#!/user/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from data_process import load_data_from_csv, seg_words, get_embeding_weights, sentences_to_indices, save_data, load_data
from data_process import convert_label_to_index, convert_index_to_label
from model import TextClassifier, build_rnn_model, trainRNNModel, predictRNNModel, load_rnn_model, predict_rnn_model
import config
import logging
import argparse
from sklearn.externals import joblib
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

    parser.add_argument('-t', '--test', type=int, nargs='?',
                        help='test mode or not')

    parser.add_argument('-lc', '--load_cache', type=int, nargs='?',
                        help='load cache or not')

    args = parser.parse_args()
    model_name = args.model_name
    if not model_name:
        model_name = "model_dict.pkl"


    is_test = True
    if args.test is None:
        is_test = True
    else:
        is_test = False if args.test == 0 else True

    load_cache = True
    if args.load_cache is None:
        load_cache = True
    else:
        load_cache = False if args.load_cache == 0 else True

    # load data
    test_num = 100 if is_test else None
    logger.info("start load data, try read {0} records, test mode {1}".format(test_num, is_test))
    test_data_df = load_data_from_csv(config.test_data_path, nrow=test_num)

    # load embedding matrix
    embedding_matrix = load_data("emb.npy")   

    # load vocab
    vocab = load_data("vocab.npy").tolist()

    # load all test columns
    columns = test_data_df.columns.tolist()

    # seg content words to sequence
    logger.info("start seg test data, let's look at some data")
    logger.info(test_data_df.iloc[1, :])
    content_test = test_data_df.iloc[:, 1]
    if not load_cache:
        sequences = data_process.sentences_to_sequence(content_test, vocab)
        save_data(sequences, "test_seq.npy")
        content_test = sequences
    else:
        content_test = load_data("test_seq.npy").tolist()

    logger.info("complete seg test data, total %s records" % len(content_test))


    # # load model
    # logger.info("start load model")
    # classifier_dict = joblib.load(config.model_save_path + model_name)

    # columns = test_data_df.columns.tolist()
    # # seg words
    # logger.info("start seg test data")
    # logger.info(test_data_df.iloc[1, :])
    # content_test = test_data_df.iloc[:, 1]
    # content_test = seg_words(content_test)
    # logger.info("complete seg test data")

    # model predict
    logger.info("start predict test data")
    for column in columns[2:]:
        
        # build rnn model and load weights
        rnn_model = build_rnn_model(data_process.VOCAB_NUMBER, embedding_matrix, data_process.NUM_CLASS)
        weights_name = column + ".h5"
        load_rnn_model(rnn_model, weights_name)

        # do prediction
        label_test = predict_rnn_model(rnn_model, content_test)
        label = data_process.convert_index_to_label(np.argmax(label_test, axis=1))
        logger.info("predict final label {0}".format(label[:100]))
        # print(len(test_data_df[column]))
        # print(len(label))
        test_data_df[column] = label
        logger.info("finish predict {0}, total {1} records".format(column, len(label)))
        if is_test:
            break

        # test_data_df[column] label_test].predict(content_test)
        logger.info("compete %s predict" % column)

    test_data_df.to_csv(config.test_data_predict_out_path, encoding="utf_8_sig", index=False)
    logger.info("compete predict test data")

# [[0.7519077  ,0.01429716, 0.95730718, 0.216488  ],[0.79128194, 0.81153971, 0.0151668,  0.18201146],[0.773996,   0.01413795, 0.01790877, 0.19395727],[0.79605305, 0.01185339, 0.01566614, 0.17642745]]


# [0.774245   0.01296784 0.01757635 0.19521077],
# [0.76460105 0.01291483 0.0161708  0.20631342],
# [0.7453211  0.0151375  0.01948236 0.22005916],
# [0.7795613  0.01302108 0.01669651 0.19072105],
# [0.7706786  0.01309566 0.01647094 0.19975486],
# [0.74976313 0.016822   0.02258777 0.21082704],
# [0.79179657 0.01191887 0.01569538 0.18058921],
# [0.70477843 0.01652352 0.02300946 0.25568852],
# [0.75692177 0.01654293 0.02186467 0.20467064],
# [0.78604305 0.01229684 0.01594508 0.18571508],
# [0.75781685 0.01262159 0.01607247 0.21348913]]
