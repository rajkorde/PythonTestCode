# -*- coding: utf-8 -*-
# https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 80
batch_size = 32

print('Loading data...')
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=max_features,
                                                      seed=113)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
