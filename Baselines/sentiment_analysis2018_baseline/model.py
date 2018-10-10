#!/user/bin/env python
# -*- coding:utf-8 -*-

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import logging

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


class TextClassifier():

    def __init__(self, vectorizer, classifier=MultinomialNB()):
        classifier = SVC(kernel="rbf")
        # classifier = SVC(kernel="linear")
        self.classifier = classifier
        self.vectorizer = vectorizer

    def features(self, x):
        return self.vectorizer.transform(x)

    def fit(self, x, y):
        features = self.features(x)
        logger.info("features shape = %s" % str(features.shape))
        self.classifier.fit(features, y)

    def predict(self, x):

        return self.classifier.predict(self.features(x))

    def score(self, x, y):
        return self.classifier.score(self.features(x), y)

    def get_f1_score(self, x, y):
        return f1_score(y, self.predict(x), average='macro')


def buildRNNModel(input_dim, embedding_weights):   # input dim in general is vocab len + 1

    output_dim = 300 

    model = Sequential()
    model.add(Embedding(input_dim, output_dim, weights=[embedding_weights], trainable = False))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    # model.layers[1].trainnable = False
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    logger.info(model.summary())

    return model

def trainRNNModel(model, content, label, name):
    logger.info("start to train....")
    train = pad_sequences(content, dtype='float32')
    model.fit(train, label, batch_size = 64, epochs = 3, verbose = 1)       # epochs to be optimized
    model.save_weights(name)
    yaml_string = model.to_yaml()
    logger.info(yaml_string)
    return model

def predictRNNModel(model, content_test, label_test):
    logger.info("start to predict....")
    X_test = pad_sequences(content_test, dtype='float32')
    score = model.evaluate(X_test, label_test, batch_size = 64)
    logger.info("predict score is %d" % score)
    return score

def load_rnn_model(name):
    logger.info("load model %s" % name)
    return load_model(name)
