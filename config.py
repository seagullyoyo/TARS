#!/usr/bin/python
# -*- coding: utf-8 -*-

class Config(object):

    version = 'bert'
    checkpoints_dir = './checkpoint'
    tensorboard_dir = './tensorboard'
    save_model_dir = './saved_model'

    vocab_file = ''
    text_train_data = 'train.txt'
    text_test_data = ''

    seq_length = 128
