#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

from config import Config
from utils.text_data_utils import text_data_process, read_text_file


class TextModel(object):
    @staticmethod
    def prepare():
        print("Loading data...")
        texts, labels = read_text_file(Config.text_train_data)
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

        return texts, labels, checkpoint, tensorboard, early_stopping, lr_decay

    def train(self, model_name):
        return

    def eval(self, model_path, test_file_path):
        return

    def predict(self):
        return