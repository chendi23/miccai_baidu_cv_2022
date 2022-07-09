# -*- coding: utf-8 -*-
"""
@Time : 2022/6/26 9:19 上午
@Auth : zcd_zhendeshuai
@File : global_variables.py
@IDE  : PyCharm

"""

import os

'''dirs'''
PROJECT_ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_ROOT_DIR = '/Users/chendi/data_competition/cv/miccai_2022_baidu/Train/'
IMG_DIR = DATA_ROOT_DIR + 'Image/'
LABEL_DIR = DATA_ROOT_DIR + 'Layer_Masks/'
TFRECORDS_DIR = PROJECT_ROOT_DIR + '/tfrecords/'
TRAIN_DIR = TFRECORDS_DIR + 'train/'
VAL_DIR = TFRECORDS_DIR + 'val/'
TEST_DIR = TFRECORDS_DIR + 'test/'
RES_DIR = PROJECT_ROOT_DIR + '/res'
CKPT_DIR = PROJECT_ROOT_DIR +'/ckpt'
PRE_TEST_DIR = '/Users/chendi/data_competition/cv/miccai_2022_baidu/Test/Image/'

'''hyperparameters'''
TOTAL_IMGS = 100
TRAIN_SET_RATIO = 0.7
ORIGINAL_IMG_SIZE = (1100, 800)
CROP_SIZE = ()
RESIZED = (1024,1024)
RESIZED_IMG = (1024,1024,3)
RESIZED_LABEL = (1024,1024,1)
CLASSES = 4
EPOCH = 1000
BATCH_SIZE = 2
LR_WARMUP = 5e-4
LR_START = 1e-3


