#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys, time

import keras.backend as K
from keras.applications.xception import Xception
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv1D, Embedding, MaxPooling1D, Flatten, concatenate
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping

PRE_TRAINED_TEXT_MODEL = ''
PRE_TRAINED_IMAGE_MODEL = ''

CHECKPOINT_DIR = ''
TENSORBOARD_PATH = ''

checkpoint = ModelCheckpoint(CHECKPOINT_DIR + 'weights.{epoch:03d}-{val_loss:.4f}.hdf5',
                             monitor='loss', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=TENSORBOARD_PATH, histogram_freq=0, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=1, mode='auto')
lr_decay = ReduceLROnPlateau(monitor='loss', patience=15, mode='auto')

# 和预训练文本配置保持一致
EMBEDDING_SIZE = 256
SEQ_LENGTH = 30


def get_text_model(vocab_size, embedding, _input):
    x = Embedding(vocab_size, EMBEDDING_SIZE, SEQ_LENGTH, weights=embedding, trainable=True, name='embedding')(_input)
    print("x={}".format(x.shape))

    conv1 = Conv1D(256, 2, padding='same', strides=1, activation='relu')(x)
    conv1 = MaxPooling1D(pool_size=29)(conv1)
    conv2 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(x)
    conv2 = MaxPooling1D(pool_size=28)(conv2)
    conv3 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(x)
    conv3 = MaxPooling1D(pool_size=28)(conv3)
    conv4 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(x)
    conv4 = MaxPooling1D(pool_size=28)(conv4)

    x = concatenate([conv1, conv2, conv3, conv4], axis=1)
    flatten = Flatten()(x)
    # l2_normalize
    x = Dropout(0.5)(flatten)
    fc = Dense(128, activation='relu')(x)
    _output = Dense(2, activation='softmax', name='output')(fc)
    model = Model(_input, _output)

    return model


def get_image_model(_input, _output):
    x = _output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=_input, outputs=predictions)

    return model


def train(pre_trained_image_model_path, pre_trained_text_model_path):
    _embedding, images, texts, labels, _vocab_size, data_size = multi_model_train_pre_process()

    _x1_train, _x2_train, _y_train, _x1_val, _x2_val, _y_val = split(images, texts, labels, data_size)

    text_input = Input(shape=(SEQ_LENGTH, ), dtype='int32', name='input')
    pre_trained_text_model = get_text_model(_vocab_size, _embedding, text_input)
    pre_trained_text_model.load_weights(pre_trained_text_model_path)
    print('pre_trained_text_model success loaded')
    names = [layer.name for layer in pre_trained_text_model.layers]
    print(names)

    pre_trained_image_model = get_image_model(base_model.input, base_model.output)
    pre_trained_image_model.load_weights(pre_trained_image_model_path)
    print('pre_trained_image_model success loaded')
    names = [layer.name for layer in pre_trained_image_model.layers]
    print(names)

    text_feat = pre_trained_text_model.get_layer('flatten_1').output
    image_feat = pre_trained_image_model.get_layer('global_average_pooling2d_1').output
    concat = concatenate([text_feat, image_feat], axis=1)
    l2_norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(concat)
    x = Dropout(0.5)(l2_norm)
    fc = Dense(128, activation='relu')(x)
    scores = Dense(2, activation='softmax', name='output')(fc)

    model = Model(inputs=[base_model.input, text_input], outputs=scores)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.summary()

    steps_per_epoch = _x1_train.shape[0] / BATCH_SIZE + 1
    validation_steps = _x1_val.shape[0] / BATCH_SIZE + 1
    model.fit_generator()