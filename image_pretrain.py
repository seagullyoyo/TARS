#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from glob import glob

import numpy as np
from PIL import ImageFile, Image
from keras.applications.xception import Xception
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn import metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True

tag = 'forbid'

TRAIN_DATA_DIR = ''
VAL_DATA_DIR = ''
TEST_DATA_DIR = ''

CKPT_PATH = ''
TENSORBOARD_PATH = ''
MODEL_PATH = ''
MODEL_NAME = ''

BATCH_SIZE = 64
EPOCH_NUM = 10
# FROZEN_BEFORE = 249  # InceptionV3
FROZEN_BEFORE = 126  # Xception

checkpoint = ModelCheckpoint(CKPT_PATH + 'weights.{epoch:03d}-{val_loss:.4f}.hdf5',
                             monitor='loss', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=TENSORBOARD_PATH, histogram_freq=0, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
lr_decay = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto')


def train(train_data_path, val_data_path, model_save_path, is_train_all):
    paths = glob(os.path.join(train_data_path, '*', '*'))

    base_model = Xception(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    train_gen = ImageDataGenerator(rescale=1. / 255, width_shift_range=0.1, height_shift_range=0.1,
                                   horizontal_flip=True)
    train_generator = train_gen.flow_from_directory(train_data_path, target_size=(299, 299), batch_size=BATCH_SIZE)

    val_gen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_gen.flow_from_directory(val_data_path, target_size=(299, 299), batch_size=BATCH_SIZE)

    steps_per_epoch = len(paths) / BATCH_SIZE + 1

    print('Start Training')
    for i, layer in enumerate(base_model.layers):
        layer.trainable = False
    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit_generator(train_generator, epochs=EPOCH_NUM, validation_data=val_generator,
                        steps_per_epoch=steps_per_epoch, callbacks=[checkpoint, tensorboard])

    print('Start FineTuning')
    if is_train_all:
        for layer in model.layers:
            layer.trainable = True
    else:
        for layer in model.layers[:FROZEN_BEFORE]:
            layer.trainable = False
        for layer in model.layers[FROZEN_BEFORE:]:
            layer.trainable = True

    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_generator, epochs=EPOCH_NUM, validation_data=val_generator,
                        steps_per_epoch=steps_per_epoch, callbacks=[checkpoint, tensorboard])

    print('Done Training')

    model.save(model_save_path + MODEL_NAME)
    print('Done Saving, model_name=' + MODEL_NAME)

    eval(model_path=model_save_path + MODEL_NAME, test_data_path=val_data_path)
    exit()


def eval(model_path, test_data_path):
    model = load_model(model_path)
    print('Loading model suucess~ model_path=' + model_path)
    test_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_gen.flow_from_directory(test_data_path, target_size=(299, 299), batch_size=BATCH_SIZE)

    prediction = model.predict_generator(test_generator)
    predict_label = np.argmax(prediction, axis=1)
    labels = test_generator.classes

    print('Precision, Recall and F1-score')
    print(metrics.classification_report(labels, predict_label, target_names=['c0', 'c1']))
    print('Confusion Matrix')
    print(metrics.confusion_matrix(labels, predict_label))
    print('Print Wrong Cases')
    index = np.arange(0, np.size(labels))
    wrong_idx = index[predict_label != labels].tolist()
    filenames = test_generator.filenames
    for idx in wrong_idx:
        print(filenames[idx])


def predict(model_path, image_path):
    model = load_model(model_path)
    print('Loading model suucess~ model_path=' + model_path)

    img = Image.open(image_path)
    img = img.resize((299, 299))
    img = np.asarray(img)

    img = img / 255.0

    img_np = img_to_array(img)
    img_np = np.expand_dims(img_np, axis=0)

    result = model.predict(img_np)
    print(result)


def load_h5_and_finetune(train_data_path, val_data_path, model_save_path, is_train_all):
    model = load_model(model_save_path)
    print('Model success loaded')

    print('Start FineTuning')
    if is_train_all:
        for layer in model.layers:
            layer.trainable = True
    else:
        for layer in model.layers[:FROZEN_BEFORE]:
            layer.trainable = False
        for layer in model.layers[FROZEN_BEFORE:]:
            layer.trainable = True

    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

    paths = glob(os.path.join(train_data_path, '*', '*'))
    steps_per_epoch = len(paths) / BATCH_SIZE + 1
    train_gen = ImageDataGenerator(rescale=1. / 255, width_shift_range=0.1, height_shift_range=0.1,
                                   horizontal_flip=True, rotation_range=40, fill_mode='nearest')
    train_generator = train_gen.flow_from_directory(train_data_path, target_size=(299, 299), batch_size=BATCH_SIZE)

    val_gen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_gen.flow_from_directory(val_data_path, target_size=(299, 299), batch_size=BATCH_SIZE)

    model.fit_generator(train_generator, epochs=EPOCH_NUM, validation_data=val_generator,
                        steps_per_epoch=steps_per_epoch, callbacks=[checkpoint, tensorboard])
    print('Saving model')
    model.save(model_save_path + '.finetune.h5')


def main():
    if sys.argv[1] == 'train':
        train(train_data_path=TRAIN_DATA_DIR, val_data_path=VAL_DATA_DIR, model_save_path=MODEL_PATH, is_train_all=True)
    if sys.argv[1] == 'eval':
        eval(model_path=MODEL_NAME, test_data_path=TEST_DATA_DIR)
    if sys.argv[1] == 'h5_finetune':
        load_h5_and_finetune(train_data_path=TRAIN_DATA_DIR, val_data_path=VAL_DATA_DIR, model_save_path=MODEL_PATH,
                             is_train_all=True)


if __name__ == '__main__':
    main()
