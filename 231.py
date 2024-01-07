import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# 加载预训练的BERT模型和配置文件
config_path = 'bert_config.json'
checkpoint_path = 'bert_model.ckpt'
tokenizer = Tokenizer('vocab.txt')
learning_rate = 1e-5
min_learning_rate = 1e-7
MAX_LEN = 100

# 加载数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_comment = train['comment'].astype(str)
test_comment = test['comment'].astype(str)
labels = train['label'].astype(int)

# 定义数据生成器
def data_generator(data):
    while True:
        idxs = np.arange(len(data[0]))
        np.random.shuffle(idxs)
        X1, Y = [], []
        for i in idxs:
            d = data[0][i]
            label = data[1][i]
            text = d[:MAX_LEN]
            x1, _ = tokenizer.encode(first=text)
            X1.append(x1)
            Y.append([label])
            if len(X1) == 32:
                yield ([np.array(X1), np.zeros_like(X1)], np.array(Y))
                X1, Y = [], []

# 定义模型
def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True
    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))
    T = bert_model([T1, T2])
    T = Lambda(lambda x: x[:, 0])(T)
    output = Dense(1, activation='sigmoid')(T)
    model = Model([T1, T2], output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

# 定义评估器
class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self, val_data, val_index):
        self.score = []
        self.best = 0.
        self.early_stopping = 0
        self.val_data = val_data
        self.val_index = val_index
        self.predict = []
        self.lr = 0
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):
        if self.passed < self.params['steps']:
            self.lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            self.lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            self.lr += min_learning_rate
            tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        score, acc, f1 = self.evaluate()
        if score > self.best:
            self.best = score
            self.early_stopping = 0
            model.save_weights('bert.w')
        else:
            self.early_stopping += 1
        print('lr: %.6f, epoch: %d, score: %.4f, acc: %.4f, f1: %.4f, best: %.4f\n' % (self.lr, epoch, score, acc, f1, self.best))

    def evaluate(self):
        self.predict = []
        prob = []
        val_x1, val_y = self.val_data
        for i in tqdm(range(len(val_x1))):
            d = val_x1[i]
            text = d[:MAX_LEN]
            t1, t1_ = tokenizer.encode(first=text)
            T1, T1_ = np.array([t1]), np.array([t1_])
            _prob = model.predict([T1, T1_])
            self.predict.append(np.argmax(_prob, axis=1)[0] + 1)
            prob.append(_prob[0])
        score = 1.0 / (1 + mean_absolute_error(val_y + 1, self.predict))
        acc = accuracy_score(val_y + 1, self.predict)
        f1 = f1_score(val_y + 1, self.predict, average='macro')
        return score, acc, f1

# 定义预测函数
def predict(data):
    prob = []
    val_x1 = data
    for i in tqdm(range(len(val_x1))):
        X = val_x1[i]
        text = X[:MAX_LEN]
        t1, t1_ = tokenizer.encode(first=text)
        T1, T1_ = np.array([t1]), np.array([t1_])
        _prob = model.predict([T1, T1_])
        prob.append(_prob[0])
    return prob

# 进行交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_test = np.zeros((len(test), 1), dtype=np.float32)
oof_train = np.zeros((len(train), 1), dtype=np.float32)
for fold, (train_index, valid_index) in enumerate(skf.split(train_comment, labels)):
    x1 = train_comment[train_index]
    y = labels[train_index]
    val_x1 = train_comment[valid_index]
    val_y = labels[valid_index]
    train_D = data_generator([x1, y])
    evaluator = Evaluate([val_x1, val_y], valid_index)
    model = get_model()
    model.fit_generator(train_D.__iter__(), steps_per_epoch=len(train_D), epochs=4, callbacks=[evaluator])
    model.load_weights('bert.w')
    oof_test += predict(test_comment)
    tf.keras.backend.clear_session()
oof_test /= 5
test['flag'] = oof_test
test['flag'] = test['flag'].apply(lambda x: 0 if x < 0.5 else 1)
test[['id', 'flag']].to_csv('bert.csv', index=False)
