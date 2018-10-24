

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

def do_validation(validate_data_df):
    
    # use RNN model to validate
    content_validate = validate_data_df.iloc[:, 1]

    logger.info("load RNN validate data sentences...")
    content_validate = data_process.sentences_to_sequence(content_validate, vocab)
    print(validate_data_df.iloc[:, 1][0])
    print(validate_data_df.iloc[:, 1][1])
    print(content_validate[0])
    print(content_validate[1])

    for column in columns[2:]:
        logger.info("start rnn validate model for %s" % column)

        # build rnn model and load weights
        rnn_model = build_rnn_model(data_process.VOCAB_NUMBER, embedding_matrix, data_process.NUM_CLASS)
        weights_name = column + ".h5"
        rnn_model_dict[column] = load_rnn_model(rnn_model, weights_name)        

        label_validate = np_utils.to_categorical(convert_label_to_index(validate_data_df[column]), num_classes = data_process.NUM_CLASS)
        logger.info(label_validate[:10])
        logger.info(validate_data_df[column][:10])
        # logger.info(label_validate[1])
        score = predictRNNModel(rnn_model_dict[column], content_validate, label_validate)
