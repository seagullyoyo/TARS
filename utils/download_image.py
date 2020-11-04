#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, platform
import time
import logging
from urllib import request
import multiprocessing
import argparse
import getpass
import sys

thread_count = 100
batch_count = 1
locker = multiprocessing.Lock()
total_file_count = 0
root_path = '/home' + getpass.getuser()
print(root_path)

DESC_DIR = "/home/allen.xg/data/images/"
IMG_FILE = "/home/allen.xg/data/image_urls/"


def get_logger(file_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(file_name + ".log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def download_image_from_url(dest_dir, URL):
    try:
        if "http" not in URL:
            URL = "http://img.taobaocdn.com/bao/uploaded/" + URL
        if sys.version_info.major == 2:
            urllib.urlretrieve(URL, os.path.join(dest_dir, URL.split('/')[-1]))
        else:
            request.urlretrieve(URL, os.path.join(dest_dir, URL.split('/')[-1]))
    except Exception as e:
        print("\tErrors retrieving the URL: ", URL, "e=", e)


def download_images_batch(urls):
    for url in urls:
        new_url = url
        download_image_from_url(DESC_DIR, new_url)

    global counter
    counter.value += len(urls)


def gen_file_queue(file_name):
    """
    生成待处理图片列表
    :param file_name:
    :return: 二维list， [[img1, img2], [img3, img4]]
    """

    urls = [url.strip() for url in open(file_name).readlines()]
    round = len(urls) / batch_count
    start = 0
    result = []
    for i in range(int(round)):
        result.append(urls[start: start + batch_count])
        start += batch_count
    result.append(urls[start:])

    global total_file_count
    total_file_count = len(urls)
    return result


def image_download(file_name, dest_dir):
    f = open(file_name)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    cnt = 0
    for url in f.readlines():
        url = url.strip()
        if "http" not in url:
            url = "http://img.taobaocdn.com/bao/uploaded/" + url
        download_image_from_url(os.path.join(dest_dir, url), url)
        cnt += 1
    print("All Done, path={}".format(dest_dir))


def init(args):
    global counter
    counter = args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--img_file", help="image file")
    parser.add_argument("-d", "--data_path", help="save image data path")
    args = parser.parse_args()

    print("dest_dir={}, img_file={}".format(DESC_DIR, IMG_FILE))

    if not os.path.exists(DESC_DIR):
        os.makedirs(DESC_DIR)
    file_list = gen_file_queue(IMG_FILE)
    counter = multiprocessing.Value('i', 0)

    pool = multiprocessing.Pool(thread_count, initializer=init, initargs=(counter,))
    run = pool.map_async(download_images_batch, file_list)
    pool.close()
    while True:
        remaning = run._number_left
        if run.ready():
            break
        print("Waiting for {} tasks to process. {} of {} files have been processed".format(remaning, counter.value, total_file_count))
        time.sleep(2)
    pool.join()
    print("All Done")
    # logger.info("All Done")
