# encoding = utf - 8

import codecs
import os
import sys

import numpy as np
import yaml
from keras import Input, Model
from keras.layers import Lambda, Dense
from keras.metrics import top_k_categorical_accuracy
from keras.optimizers import Adam
from keras.models import load_model

from keras_bert import Tokenizer, load_trained_model_from_checkpoint
from sklearn import metrics

from config import Config
from model.text_model import TextModel
from utils.text_data_utils import text_data_process

config_path = 'AdBertBase/bert_config.json'
checkpoint_path = 'AdBertBase/model.ckpt'
dict_path = 'AdBertBase/vocab.txt'


class BertModel(TextModel):
    def __init__(self):
        self.model = self.init()
        self.token_dict = get_dict()
        self.tokenizer = OurTokenizer(self.token_dict)

    @staticmethod
    def init():
        bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

        for l in bert_model.layers:
            l.trainable = False

        input_words = Input(shape=(None,), name='input_words')
        input_pos = Input(shape=(None,), name='input_pos')

        x = bert_model([input_words, input_pos])
        x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
        p = Dense(2, activation='softmax')(x)

        model = Model([input_words, input_pos], p)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(1e-5),  # 用足够小的学习率
                      metrics=['accuracy', acc_top3])
        print(model.summary())
        return model

    def train(self, base_model=None):
        texts, labels, checkpoint, tensorboard, early_stopping, lr_decay = self.prepare()

        train_texts = texts[:int(len(labels) * 0.95)]
        val_texts = texts[int(len(labels) * 0.95):]
        train_labels = labels[:int(len(labels) * 0.95)]
        val_labels = labels[int(len(labels) * 0.95):]

        train_gen = BertDataGenerator(train_texts, train_labels, self.tokenizer, shuffle=True)
        valid_gen = BertDataGenerator(val_texts, val_labels, self.tokenizer)

        # 模型训练
        self.model.fit(
            train_gen.__iter__(),
            steps_per_epoch=len(train_gen),
            epochs=5,
            validation_data=valid_gen.__iter__(),
            validation_steps=len(valid_gen),
            callbacks=[checkpoint, tensorboard, early_stopping, lr_decay],
        )

        print("Saving model...")
        if not os.path.exists(Config.save_model_dir):
            os.mkdir(Config.save_model_dir)
        model_name = os.path.join(Config.save_model_dir, Config.version) + '.h5'
        self.model.save(model_name)
        print('Model saved to {}'.format(model_name))
        print(self.model.summary())

    def eval(self, model_path, test_file_path):
        self.model = load_model(model_path)
        texts, labels = text_data_process(Config.text_test_data)

        predicts = self.model.predict({'input_words': texts})
        predicts = np.argmax(predicts, axis=1)
        labels = np.argmax(labels, axis=1)

        print('Precision, Recall and F1-score')
        print(metrics.classification_report(labels, predicts))
        print('Confusion Matrix')
        print(metrics.confusion_matrix(labels, predicts))
        print('Print Wrong Cases')
        index = np.arange(0, np.size(labels))
        wrong_idx = index[predicts != labels].tolist()
        for idx in wrong_idx:
            print(texts[idx])
        return


class BertDataGenerator:
    def __init__(self, texts, labels, tokenizer, batch_size=32, shuffle=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.labels) // self.batch_size
        if len(self.labels) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.labels)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                text = self.texts[i][:Config.seq_length]
                token_ids, segment_ids = self.tokenizer.encode(first=text)
                label = self.labels[i]
                X1.append(token_ids)
                X2.append(segment_ids)
                Y.append([label])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示
        return R


def get_dict():
    # 将词表中的词编号转换为字典
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def acc_top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
