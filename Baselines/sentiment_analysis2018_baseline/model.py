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
from keras import backend as K

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

BATCH_SIZE = 1024 # 200000

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

# class Metrics(Callback):
#     def on_train_begin(self, logs={}):
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []

#     def on_epoch_end(self, epoch, logs={}):
#         val_predict = (numpy.asarray(self.model.predict(
#             self.validation_data[0]))).round()
#         val_targ = self.validation_data[1]
#         _val_f1 = f1_score(val_targ, val_predict)
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
#         self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         return

def build_rnn_model(input_dim, embedding_weights, num_class):   # input dim in general is vocab len + 1

    output_dim = 300 
    # max_input_length = 1024

    model = Sequential()
    model.add(Embedding(input_dim, output_dim, weights=[embedding_weights], trainable = False))
    model.add(LSTM(32))
    # model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Dense(num_class))
    model.add(Activation('softmax'))

    # model.layers[1].trainnable = False
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1]) # , f1, recall, precision])

    logger.info(model.summary())

    return model

def trainRNNModel(model, content, label, name, epochs):

    # metrics = Metrics()

    logger.info("start to train....")
    train = pad_sequences(content, dtype='float32')
    model.fit(train, label, batch_size = BATCH_SIZE, epochs = epochs, verbose = 1)       # epochs to be optimized
    model.save_weights(name)
    yaml_string = model.to_yaml()
    logger.info(yaml_string)
    logger.info("save model weights %s" % name)
    return model

def predictRNNModel(model, content_test, label_test):
    logger.info("start to predict....")
    X_test = pad_sequences(content_test, dtype='float32')
    score = model.evaluate(X_test, label_test, batch_size = BATCH_SIZE)
    logger.info("predict score is %s" % score)
    # Y_pred = model.predict(X_test, batch_size=64, verbose=1)
    
    # print(Y_pred)
    # print(label_test)
    
    # score1 = accuracy_score(label_test.tolist(), Y_pred.tolist())
    # score2 = f1_score(label_test.tolist(), Y_pred.tolist())
    # logger.info("acc : %s, f1 : %s" % score1, score2)
    return score

def predict_rnn_model(model, content_test):
    logger.info("start to predict....")
    X_test = pad_sequences(content_test, dtype='float32')
    Y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    
    logger.info("predict done, some result here {0}".format(Y_pred[:10]))
    # print(label_test)
    
    # score1 = accuracy_score(label_test.tolist(), Y_pred.tolist())
    # score2 = f1_score(label_test.tolist(), Y_pred.tolist())
    # logger.info("acc : %s, f1 : %s" % score1, score2)
    # score = model.evaluate(X_test, label_test, batch_size = 64)
    # logger.info("predict score is %s" % score)
    # print(score)
    return Y_pred


def load_rnn_model(model, name):
    logger.info("load model weights %s" % name)
    model.load_weights(name)
    return model

def mcor(y_true, y_pred):
     # matthews_correlation
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
 
 
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
 
 
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
 
 
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
 
 
     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
 
 
     return numerator / (denominator + K.epsilon())




def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
