# -*- coding: utf-8 -*-

import keras
from PIL import Image
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input as VGG19_process
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.misc import toimage

print(keras.__version__)


def load_original(img_path):
    image = Image.open(img_path)
    image = image.convert("RGB")
    img = image.resize(224, 224)
    plt.figure(0)
    plt.subplot(131)
    plt.imshow(np.asarray(img))
    return img


def load_trained_model(img):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    net = base_model.output
    net = GlobalAveragePooling2D()(net)
    net = Dense(1024, activation='relu')(net)
    predicitons = Dense(1000, activation='softmax')(net)
    model = Model(inputs=base_model.input, outputs=predicitons)
    model.load_weights('./GeM_VGG19_2019.weights.hdf5')

    print(model.summary())

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = VGG19_process(x)

    return model, x


def extract_features(ins, layer_id, filters, layer_num):
    """
    提取模型指定层指定数目的 feature map 并输出到一幅图上
    :param ins: 模型实例
    :param layer_id: 提取指定层特征
    :param filters: 每层提取的 feature map 数
    :param layer_num: 一共提取多少层 feature map
    :return: None
    """
    if len(ins) != 2:
        print("parameter error: (model, instance)")
        return None
    model = ins[0]
    x = ins[1]
    if type(layer_id) == type(1):
        model_extractfeatures = Model(input=model.input, output=model.get_layer(index=layer_id).output)
    else:
        model_extractfeatures = Model(input=model.input, output=model.get_layer(name=layer_id).output)
    fc2_features = model_extractfeatures.predict(x)
    if filters > len(fc2_features[0][0][0]):
        print("layer number error.", len(fc2_features[0][0][0]), ',', filters)
        return None
    for i in range(filters):
        plt.subplot_adjust(left=0, right=1, bottom=0, top=1)
        plt.subplot(filters, layer_num, layer_id + 1 + i * layer_num)
        plt.axis("off")
        if i < len(fc2_features[0][0][0]):
            plt.imshow(fc2_features[0, :, :, i])


# 层数、模型、卷积核数
def extract_features_batch(layer_num, model, filters):
    """
    批量提取特征
    :param layer_num:层数
    :param model: 模型
    :param filters: feature map 数
    :return: None
    """
    plt.figure(figsize=(filters, layer_num))
    plt.subplot(filters, layer_num, 1)
    for i in range(1, layer_num):
        extract_features(model, i, filters, layer_num)
    plt.show()


def extract_features_with_layers(layer_extract):
    """
    提取hypercolumn并可视化
    :param layer_extract: 指定层列表
    :return: None
    """
    hc = extract_hypercolumn(x[0], layer_extract, x[1])
    ave = np.average(hc.transpose(1, 2, 0), axis=2)
    hc_max = np.max(hc.transpose(1, 2, 0), axis=2)
    plt.subplot(132)
    plt.imshow(ave)
    plt.subplot(133)
    plt.imshow(hc_max)
    plt.show()


def extract_hypercolumn(model, layer_indexes, instance):
    """
    提取指定模型指定层的hypercolumn向量
    :param model: 模型
    :param layer_indexes: 层id
    :param instance: 模型
    :return:
    """
    feature_maps = []
    for i in layer_indexes:
        feature_maps.append(Model(input=model.input, output=model.get_layer(index=i).output).predict(instance))
    hypercolumns = []
    for convmap in feature_maps:
        for i in range(convmap[0][0][0].shape[0]):
            unscaled = sp.misc.imresize(convmap[0, :, :, i], size=(224, 224), mode="F", interp="bilinear")
            hypercolumns.append(unscaled)
    return np.asarray(hypercolumns)


if __name__ == '__main__':
    img_path = './data/4.jpeg'

    img = load_original(img_path)
    x = load_trained_model(img)
    # extract_features_batch(20, x, 3)
    # extract_features_with_layers([1, 3, 5])
    # extract_features_with_layers([1, 2, 4, 5, 8, 10, 15, 19, 20])
    extract_features_with_layers([20])

    print("done")
