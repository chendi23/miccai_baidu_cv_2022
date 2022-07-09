# -*- coding: utf-8 -*-
"""
@Time : 2022/6/26 9:29 上午
@Auth : zcd_zhendeshuai
@File : preprocessing.py
@IDE  : PyCharm

"""

import os

import numpy as np
import tensorflow as tf
import skimage
import cv2
from sklearn.model_selection import train_test_split
from common_utils import parser_args, get_Float_ListFeature, serialize_array, bytes_feature, MyMIOU, parse_example
from config import global_variables as gl


class Pre_processing:
    def __init__(self):
        if not os.path.exists(gl.TFRECORDS_DIR):
            os.mkdir(gl.TFRECORDS_DIR)
            os.mkdir(gl.TRAIN_DIR)
            os.mkdir(gl.VAL_DIR)
            os.mkdir(gl.TEST_DIR)
            os.mkdir(gl.CKPT_DIR)
            os.mkdir(gl.RES_DIR)
        params_dict = {}
        params_dict['epoch'] = gl.EPOCH
        params_dict['batch_size'] = gl.BATCH_SIZE
        params_dict['lr_warmup'] = gl.LR_WARMUP
        params_dict['lr_start'] = gl.LR_START
        self.args = parser_args(params_dict)

    def get_split_img_label_dirs(self):
        relative_dirs = os.listdir(gl.IMG_DIR)
        n = len(relative_dirs)
        if '.DS_Store' in relative_dirs:
            n = n - 1
        split = train_test_split(range(n), train_size=0.7, test_size=0.3)
        train_index, val_index = split[0], split[1]
        split_train_dirs, split_val_dirs = [], []
        for i in train_index:
            real_i = i + 1
            if real_i < 10:
                img_dir = os.path.join(gl.IMG_DIR, '000%d.png' % real_i)
                label_dir = os.path.join(gl.LABEL_DIR, '000%d.png' % real_i)
            elif 10 <= real_i < 100:
                img_dir = os.path.join(gl.IMG_DIR, '00%d.png' % real_i)
                label_dir = os.path.join(gl.LABEL_DIR, '00%d.png' % real_i)
            else:
                img_dir = os.path.join(gl.IMG_DIR, '0%d.png' % real_i)
                label_dir = os.path.join(gl.LABEL_DIR, '0%d.png' % real_i)
            split_train_dirs.append((img_dir, label_dir))

        for i in val_index:
            real_i = i + 1
            if real_i < 10:
                img_dir = os.path.join(gl.IMG_DIR, '000%d.png' % real_i)
                label_dir = os.path.join(gl.LABEL_DIR, '000%d.png' % real_i)
            elif 10 <= real_i < 100:
                img_dir = os.path.join(gl.IMG_DIR, '00%d.png' % real_i)
                label_dir = os.path.join(gl.LABEL_DIR, '00%d.png' % real_i)
            else:
                img_dir = os.path.join(gl.IMG_DIR, '0%d.png' % real_i)
                label_dir = os.path.join(gl.LABEL_DIR, '0%d.png' % real_i)
            split_val_dirs.append((img_dir, label_dir))

        return split_train_dirs, split_val_dirs

    def read_dirs_to_tensor(self, dir):
        array = skimage.io.imread(dir)
        return array

    def make_tfrecords(self):
        split_train_dirs, split_val_dirs = self.get_split_img_label_dirs()
        with tf.io.TFRecordWriter(path=os.path.join(gl.VAL_DIR, 'val_0.tfrecords')) as wr:
            for i in split_val_dirs:
                img, label = self.read_dirs_to_tensor(i[0]), self.read_dirs_to_tensor(i[1])
                image_resized = cv2.resize(img, gl.RESIZED, interpolation=cv2.INTER_LINEAR) / 255.
                label[label == 0] = 1
                label[label == 80] = 2
                label[label == 160] = 3
                label[label == 255] = 0
                label_resized = cv2.resize(label, gl.RESIZED)
                label_resized = label_resized[:, :, 1]
                img_feature = bytes_feature(serialize_array(image_resized))
                label_feature = bytes_feature(serialize_array(label_resized))
                single_row_dict = {}
                single_row_dict['image'] = img_feature
                single_row_dict['label'] = label_feature
                features = tf.train.Features(feature=single_row_dict)
                example = tf.train.Example(features=features)
                wr.write(record=example.SerializeToString())
            wr.close()
        with tf.io.TFRecordWriter(path=os.path.join(gl.TRAIN_DIR, 'train_0.tfrecords')) as wr:
            for i in split_train_dirs:
                img, label = self.read_dirs_to_tensor(i[0]), self.read_dirs_to_tensor(i[1])
                image_resized = cv2.resize(img, gl.RESIZED, interpolation=cv2.INTER_LINEAR) / 255.
                label[label == 0] = 1
                label[label == 80] = 2
                label[label == 160] = 3
                label[label == 255] = 0
                label_resized = cv2.resize(label, gl.RESIZED)
                label_resized = label_resized[:, :, 1]
                img_feature = bytes_feature(serialize_array(image_resized))
                label_feature = bytes_feature(serialize_array(label_resized))
                single_row_dict = {}
                single_row_dict['image'] = img_feature
                single_row_dict['label'] = label_feature
                features = tf.train.Features(feature=single_row_dict)
                example = tf.train.Example(features=features)
                wr.write(record=example.SerializeToString())
            wr.close()

    def make_test_records(self):

        relative_dirs = os.listdir(gl.PRE_TEST_DIR)
        n = len(relative_dirs)
        if '.DS_Store' in relative_dirs:
            n = n - 1
        test_dirs = []
        for i in range(n):
            real_i = i + 101
            img_dir = os.path.join(gl.PRE_TEST_DIR, '0%d.png' % real_i)
            test_dirs.append(img_dir)
        with tf.io.TFRecordWriter(path=os.path.join(gl.TEST_DIR, 'pre_test_0.tfrecords')) as wr:
            for i in test_dirs:
                img = self.read_dirs_to_tensor(i)
                image_resized = cv2.resize(img, gl.RESIZED, interpolation=cv2.INTER_LINEAR) / 255.
                img_feature = bytes_feature(serialize_array(image_resized))
                single_row_dict = {}
                single_row_dict['image'] = img_feature
                features = tf.train.Features(feature=single_row_dict)
                example = tf.train.Example(features=features)
                wr.write(record=example.SerializeToString())
            wr.close()
    def check_tfrecords(self, split_mode='val', index=0):
        import random

        filename = gl.TFRECORDS_DIR + split_mode + '/' + '{0}_{1}.tfrecords'.format(split_mode, index)
        ds = tf.data.TFRecordDataset(filename)
        ds = ds.map(lambda x: parse_example(x)).prefetch(buffer_size=10).batch(self.args.batch_size, drop_remainder=True)
        # itr = ds.make_one_shot_iterator()
        # batch_data = itr.get_next()
        for img, lb in ds:
            print('batched image shape: ', img.shape)
            print('batched label shape: ', lb.shape)

        random_index = random.randint(0, self.args.batch_size - 1)
        proto_img, proto_lb = tf.make_tensor_proto(img[random_index]), tf.make_tensor_proto(lb[random_index])
        sampled_image, sampled_label = tf.make_ndarray(proto_img), tf.make_ndarray(proto_lb)
        skimage.io.imshow(sampled_image)
        skimage.io.show()
        skimage.io.imshow(sampled_label)
        print('num label:', len(np.unique(sampled_label)))
        skimage.io.show()


def main():
    prepare = Pre_processing()
    prepare.make_tfrecords()
    prepare.make_test_records()
    prepare.check_tfrecords('val')


if __name__ == '__main__':
    main()
