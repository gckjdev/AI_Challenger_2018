#!/user/bin/env python
# -*- coding:utf-8 -*-

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

    args = parser.parse_args()
    model_name = args.model_name
    if not model_name:
        model_name = "model_dict.pkl"


    is_test = True
    if args.test is None:
        is_test = True
    else:
        is_test = False if args.test == 0 else True

    # load data
    logger.info("start load data")
    test_num = None
    test_data_df = load_data_from_csv(config.test_data_path, nrow=test_num)

    # load embedding matrix
    embedding_matrix = load_data("emb.npy")

    

    # load vocab
    vocab = load_data("vocab.npy").tolist()

    # load all test columns
    columns = test_data_df.columns.tolist()

    # seg content words to sequence
    logger.info("start seg test data")
    logger.info(test_data_df.iloc[1, :])
    content_test = test_data_df.iloc[:, 1]
    content_test = data_process.sentences_to_sequence(content_test, vocab)
    logger.info("complete seg test data")


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
        test_data_df[column] = predict_rnn_model(rnn_model, content_test)
        if is_test:
            break

        # test_data_df[column] = classifier_dict[column].predict(content_test)
        logger.info("compete %s predict" % column)

    test_data_df.to_csv(config.test_data_predict_out_path, encoding="utf_8_sig", index=False)
    logger.info("compete predict test data")
