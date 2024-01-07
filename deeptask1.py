# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 22:41:37 2022

@author: Lenovo
"""
import re
import random
import codecs,gc
import jieba
from gensim.models import Word2Vec,FastText
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import *
from keras.callbacks import *
from tensorflow.keras.utils import to_categorical 
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Model 
from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential
from keras.backend import set_session
from gensim.models import KeyedVectors
import torch.nn.functional as F
import gensim
import numpy as np
import pandas as pd
import os
import pickle
from torch.optim.optimizer import Optimizer
tf.compat.v1.disable_eager_execution()


class AdamW(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/4)
                 epsilon=1e-8, decay=0., **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            # decoupled weight decay (2/4)
            self.wd = K.variable(weight_decay, name='weight_decay')
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd  # decoupled weight decay (3/4)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            # decoupled weight decay (4/4)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.wd)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


train = pd.read_csv('../input/deepdataset/train.csv', sep='\t')
test = pd.read_csv('../input/deepdataset/test_new.csv', sep=',')


def filter(comment):
    comment = re.sub(r"[!\#\$\%\&\'\,\-\.\/\:\;\<\=\>\?\@\^\_\`\~\“\”\？\，\！\《\》\【\】\（\）\、\。\：\；\’\‘\……\￥\·\"\"\' ']",
                     '', comment)
    comment = comment.replace('图片', '')
    comment = comment.replace('\xa0', '')
    return comment


train['comment'] = train['comment'].map(lambda x: filter(x))
test['comment'] = test['comment'].map(lambda x: filter(x))
stop_words = pd.read_table('../input/deepdataset/word.txt', encoding='UTF-8', header=None)[0].tolist()
train['comment'] = train['comment'].map(lambda x: [s for s in list(jieba.cut(x)) if s not in stop_words])
test['comment'] = test['comment'].map(lambda x: [s for s in list(jieba.cut(x)) if s not in stop_words])

totaldata = pd.concat([train, test])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(totaldata['comment'])
word_index = tokenizer.word_index
index_word = tokenizer.index_word
print('已将词转换为序号')
x_train_word_ids = tokenizer.texts_to_sequences(train['comment'])
x_test_word_ids = tokenizer.texts_to_sequences(test['comment'])
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=100)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=100)

w2v_model = Word2Vec(window=2, min_count=1, workers=5, sg=0)
w2v_model.build_vocab(totaldata['comment'])
w2v_model.train(totaldata['comment'], total_examples=w2v_model.corpus_count, epochs=50)
embedding_matrix1 = np.zeros((len(word_index) + 1, 100))

print('成功生成词向量1')

fast_model = FastText(window=2, min_count=1, workers=5, sg=0)
fast_model.build_vocab(totaldata['comment'])
fast_model.train(totaldata['comment'], total_examples=fast_model.corpus_count, epochs=50)
embedding_matrix2 = np.zeros((len(word_index) + 1, 100))

print('成功生成词向量2')

for word, i in word_index.items():
    try:
        embedding_vector = w2v_model.wv[str(word)]
        embedding_matrix1[i] = embedding_vector
    except KeyError:
        continue

for word, i in word_index.items():
    try:
        embedding_vector = fast_model.wv[str(word)]
        embedding_matrix2[i] = embedding_vector
    except KeyError:
        continue

embedding_matrix = np.concatenate([embedding_matrix1, embedding_matrix2])

# Convolution  卷积
filter_length = 5  # 滤波器长度
nb_filter = 64  # 滤波器个数
# LSTM
lstm_output_size1 = 256  # LSTM 层输出尺寸
lstm_output_size2 = 128
# Training   训练参数
batch_size = 30  # 批数据量大小
nb_epoch = 4  # 迭代次数

num_filter = 100
kernel_sizes = 5


def bilstm_CNN_model(embedding_matrix):
    modelTF = tf.keras.Sequential([
        tf.keras.layers.Embedding(embedding_matrix.shape[0], 100, weights=[embedding_matrix]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv1D(num_filter, kernel_sizes, padding='valid', activation='relu', strides=1),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    modelTF.compile(loss='binary_crossentropy',
                    optimizer=AdamW(1e-4),
                    metrics=['accuracy'])
    return modelTF


x_data = x_train_padded_seqs
y_data = train['label'].tolist()
y_data = np.array(y_data)
test_preds1 = np.zeros(len(test['comment'].tolist()))
test_preds2 = np.zeros((len(test['comment'].tolist()), 1))

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1017).split(x_train_padded_seqs, y_data)
for i, (train_index, valid_index) in enumerate(kf):
    x_train, y_train = x_data[train_index], y_data[train_index]
    x_valid, y_valid = x_data[valid_index], y_data[valid_index]
    model2 = bilstm_CNN_model(embedding_matrix)

    early_stop = tf.keras.callbacks.EarlyStoppingg(
        monitor='val_loss',
        patience=2,
        verbose=0,
        mode='auto',
        restore_best_weights=True)

    print('第{}次交叉验证'.format(i + 1))
    history = model2.fit(x_train, y_train, batch_size=50, validation_data=(x_valid, y_valid),
                         epochs=15)
    plot_graphs(history, 'accuracy')
    test_preds2 += model2.predict(x_test_padded_seqs) / 5
sub = test.copy()
x = test_preds2.tolist()
res = []
for i in range(len(x)):
    if x[i][0] >= 0.5:
        res.append(1)
    else:
        res.append(0)
sub['label'] = res
sub[['id', 'label']].to_csv('val.csv')