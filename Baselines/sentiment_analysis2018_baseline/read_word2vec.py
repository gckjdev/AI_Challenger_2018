
import numpy as np
import argparse
import random
import codecs
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
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
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i

    logger.info("Word2Vec loading.... total %s words read" % len(iw))

    return vectors, iw, wi, dim