# -*- coding: utf-8 -*-
"""
@Time : 2022/6/26 9:39 上午
@Auth : zcd_zhendeshuai
@File : common_utils.py
@IDE  : PyCharm

"""

import tensorflow as tf

K = tf.keras.backend
import numpy as np
import skimage
from config import global_variables as gl
from argparse import ArgumentParser


def parser_args(params_dict):
    parser = ArgumentParser()
    for k, v in params_dict.items():
        parser.add_argument('--%s' % k, default=v)

    args = parser.parse_args()
    return args


def get_Float_ListFeature(value):
    if not isinstance(value, np.ndarray):
        value = np.asarray(value)
        value = value.astype(np.float16).tostring()
        value = [value]
        float_list = tf.train.BytesList(value=value)
        return tf.train.Feature(bytes_list=float_list)
    else:
        value = value.astype(np.float32).tostring()
        value = [value]
        float_list = tf.train.BytesList(value=value)
        return tf.train.Feature(bytes_list=float_list)


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_example(example, split='train'):
    if split in ['train', 'val']:
        expected_features = {'image': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                             'label': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                             }
        parsed_features = tf.io.parse_single_example(serialized=example, features=expected_features)
        image, label = parsed_features['image'], parsed_features['label']
        image, label = tf.io.parse_tensor(image, out_type=tf.float64), tf.io.parse_tensor(label,
                                                                                          tf.uint8)
        image = tf.reshape(image, gl.RESIZED_IMG)
        label = tf.reshape(label, gl.RESIZED_LABEL)
        label = tf.reshape(label, gl.RESIZED_LABEL)

    else:
        expected_features = {'image': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                             }
        parsed_features = tf.io.parse_single_example(serialized=example, features=expected_features)
        image = parsed_features['image']
        image = tf.io.parse_tensor(image, out_type=tf.float64)
        image = tf.reshape(image, gl.RESIZED_IMG)
        return image

    return image, label


def get_file_list(input_path):
    file_list = tf.compat.v1.gfile.ListDirectory(input_path)
    file_dir_list = []
    for i in file_list:
        file_dir_list.append(input_path + '/' + i)
    print('number of tfrecords files:', len(file_dir_list))
    return file_dir_list


'''convert rgb label to grey'''


def rgb2mask(img):
    cmap = {(255, 255, 0): 0, (0, 255, 255): 1, (255, 255, 255): 2}
    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3

    W = np.power(256, [[0], [1], [2]])

    img_id = img.dot(W).squeeze(-1)
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)

    for c in enumerate(values):
        try:
            mask[img_id == c] = cmap[tuple(img[img_id == c][0])]
        except:
            pass
    return mask


class MyMIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int64)
        return super().update_state(tf.squeeze(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


class MyCosSim(tf.keras.metrics.CosineSimilarity):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int64)
        return super().update_state(tf.squeeze(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

def ce(y_true,y_pred, from_logits=False):
    loss = tf.keras.losses.sparse_categorical_crossentropy(from_logits=from_logits, y_true=y_true,
                                                            y_pred=y_pred)
    return loss

def dice_coef(y_true, y_pred, smooth=1e-7, from_logits=True):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    if from_logits:
        y_pred = K.softmax(y_pred, -1)
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=4)[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    # y_pred_f = K.flatten(K.one_hot(y_pred,num_classes=4)[..., 1:])

    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)

    return K.mean((2. * intersect / (denom + smooth)))

    # ###for task1###
    # y_pred_RNFL = tf.cast(tf.equal(y_pred, 1), tf.int32)
    # y_pred_GCIPL = tf.cast(tf.equal(y_pred, 2), tf.int32)
    # y_pred_choroid = tf.cast(tf.equal(y_pred, 3), tf.int32)
    #
    # y_true_RNFL = tf.cast(tf.equal(y_true, 1), tf.int32)
    # y_true_GCIPL = tf.cast(tf.equal(y_true, 2), tf.int32)
    # y_true_choroid = tf.cast(tf.equal(y_true, 3), tf.int32)
    #
    #
    # y_pred_f_RNFL = K.flatten(K.one_hot(K.cast(y_pred_RNFL, 'int32'), num_classes=2)[...,1:])
    # y_pred_f_GCIPL = K.flatten(K.one_hot(K.cast(y_pred_GCIPL, 'int32'), num_classes=2)[...,1:])
    # y_pred_f_choroid = K.flatten(K.one_hot(K.cast(y_pred_choroid, 'int32'), num_classes=2)[...,1:])
    #
    # y_true_f_RNFL = K.flatten(K.one_hot(K.cast(y_true_RNFL, 'int32'), num_classes=2)[...,1:])
    # y_true_f_GCIPL = K.flatten(K.one_hot(K.cast(y_true_GCIPL, 'int32'), num_classes=2)[...,1:])
    # y_true_f_choroid = K.flatten(K.one_hot(K.cast(y_true_choroid, 'int32'), num_classes=2)[...,1:])
    #
    # intersect_RNFL = K.sum(y_true_f_RNFL*y_pred_f_RNFL,axis=-1)
    # denom_RNFL = K.sum(y_true_f_RNFL+y_pred_f_RNFL, axis=-1)
    #
    # intersect_GCIPL = K.sum(y_true_f_GCIPL*y_pred_f_GCIPL,axis=-1)
    # denom_GCIPL = K.sum(y_true_f_GCIPL+y_pred_f_GCIPL, axis=-1)
    #
    # intersect_choroid = K.sum(y_true_f_choroid*y_pred_f_choroid,axis=-1)
    # denom_choriod = K.sum(y_true_f_choroid+y_pred_f_choroid, axis=-1)
    #
    # RNFL_DICE = K.mean((2. * intersect_RNFL / (denom_RNFL + smooth)))
    # GCIPL_DICE = K.mean((2. * intersect_GCIPL / (denom_GCIPL + smooth)))
    # choroid_DICE = K.mean((2. * intersect_choroid / (denom_choriod + smooth)))
    #
    # return 0.3*RNFL_DICE+0.4*GCIPL_DICE+0.3*choroid_DICE


def dice_ce(y_true, y_pred, from_logits=True):
    loss = tf.keras.losses.sparse_categorical_crossentropy(from_logits=from_logits, y_true=y_true,
                                                           y_pred=y_pred) + dice_loss(y_true, y_pred, from_logits)
    return loss


def dice_loss(y_true, y_pred, from_logts=True):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''

    loss = 1 - dice_coef(y_true, y_pred, from_logits=from_logts)
    return loss


def omit_ce(y_true, y_pred, from_logits=False):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, 0))
    if from_logits:
        y_pred = K.softmax(y_pred, -1)
    y_pred_masked = tf.boolean_mask(y_pred, tf.squeeze(tf.not_equal(y_true, 0), -1))
    loss = tf.keras.losses.sparse_categorical_crossentropy(from_logits=from_logits,
                                                           y_true=tf.cast(y_true_masked, tf.float32),
                                                           y_pred=y_pred_masked)
    return loss


def dice_omit_ce(y_true, y_pred, from_logit=False):
    dice_l = dice_loss(y_true, y_pred, from_logit)
    ce_l = omit_ce(y_true, y_pred, from_logit)
    return dice_l + ce_l


def mean_iou(y_true, y_pred):
    y_true = K.one_hot(y_true, num_classes=4)
    y_true = tf.argmax(y_true, axis=-1)
    y_true = tf.cast(y_true, tf.dtypes.float64)
    y_pred = tf.cast(y_pred, tf.dtypes.float64)
    I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
    U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
    return tf.reduce_mean(I / U)


def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice
