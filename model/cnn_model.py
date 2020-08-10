#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, SpatialDropout1D, Conv1D, MaxPooling1D, concatenate, Flatten, Dropout, Dense
from keras.models import load_model
from keras.optimizers import Nadam
from sklearn import metrics

from config import Config
from model.text_model import TextModel


# 不分词
from utils.text_data_utils import text_data_process


class CnnModel(TextModel):
    def __init__(self, embedding_matrix, vocab_size, embedding_dim):
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.model = self.init()

    def init(self):
        input_words = Input(shape=(Config.seq_length,), dtype='int32', name='input_words')
        x = Embedding(self.vocab_size, self.embedding_dim, input_length=Config.seq_length,
                      weights=[self.embedding_matrix], trainable=True, name='word_embedding')(input_words)
        x = SpatialDropout1D(0.5)(x)
        conv1 = Conv1D(256, 2, padding='same', strides=1, activation='relu')(x)
        pool1 = MaxPooling1D(pool_size=(int(Config.seq_length) - 1))(conv1)
        conv2 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(x)
        pool2 = MaxPooling1D(pool_size=(int(Config.seq_length) - 2))(conv2)
        conv3 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(x)
        pool3 = MaxPooling1D(pool_size=(int(Config.seq_length) - 3))(conv3)
        conv4 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(x)
        pool4 = MaxPooling1D(pool_size=(int(Config.seq_length) - 4))(conv4)

        x = concatenate([pool1, pool2, pool3, pool4], axis=1)
        x = Flatten(name='flatten')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(2, activation='softmax', name='scores')(x)
        model = Model(input_words, output)
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        print(model.summary())
        return model

    def train(self, base_model=None):
        if base_model is not None:
            print("Loading pretrained model...")
            self.model = load_model(base_model)
            self.model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-5), metrics=['accuracy'])
        print("Loading data...")
        texts, labels = text_data_process(Config.text_train_data)
        # data = list(zip(texts, labels))
        # train_data = data[:len(data) * 0.95]
        # val_data = data[len(data) * 0.95:]

        checkpoint_dir = os.path.join(Config.checkpoints_dir, Config.version)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint = ModelCheckpoint(filepath=checkpoint_dir + '/weights.{epoch:03d}-{val_acc:4f}.hdf5',
                                     monitor='accuracy', verbose=1, save_best_only=True)

        tensorboard_dir = os.path.join(Config.tensorboard_dir, Config.version)
        if not os.path.exists(tensorboard_dir):
            os.mkdir(tensorboard_dir)

        tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)

        early_stopping = EarlyStopping(monitor='train_loss', patience=10, verbose=1)
        lr_decay = ReduceLROnPlateau(monitor='train_loss', patience=5, min_lr=1e-6)

        self.model.fit(x=texts, y=labels, validation_split=0.95, batch_size=128, epochs=10,
                       callbacks=[checkpoint, tensorboard, early_stopping, lr_decay])

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
        print(metrics.classification_report(labels, predicts, target_names=['c0', 'c1']))
        print('Confusion Matrix')
        print(metrics.confusion_matrix(labels, predicts))
        print('Print Wrong Cases')
        index = np.arange(0, np.size(labels))
        wrong_idx = index[predicts != labels].tolist()
        for idx in wrong_idx:
            print(texts[idx])
        return

    def predict(self):
        return
