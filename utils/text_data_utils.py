#!/usr/bin/python
# -*- coding: utf-8 -*-
import io
import os
import keras as kr
import numpy as np
from keras_preprocessing import text, sequence
from pickle import dump, load

from config import Config


def read_text_file(filename, remove_blank=False):
    texts = []
    labels = []
    with io.open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line.split('\t')) != 2:
                continue
            text, label = line.split('\t')
            if remove_blank:
                text = text.replace(' ', '')
            texts.append(text)
            labels.append(label)
    assert (len(texts) == len(labels))
    return texts, labels


def text_data_process(data_file):
    tonkenizer_char = load(open(Config.vocab_file, 'rb'))
    print("tokenizer loaded")

    if not os.path.exists(data_file):
        print("{} not found".format(data_file))
        exit()

    texts, labels = read_text_file(data_file, remove_blank=True)

    indices = np.random.permutation(len(labels))
    texts = np.array(texts)[indices]
    labels = np.array(labels)[indices]

    text_ids = np.array(tonkenizer_char.texts_to_sequences(texts=texts))
    text_ids = sequence.pad_sequences(text_ids, maxlen=Config.seq_length, padding='post', truncating='post')

    labels_ids = kr.utils.to_categorical(labels, num_classes=2)
    print("y = {} while label = {}".format(labels_ids[0], labels[0]))

    return text_ids, labels_ids
