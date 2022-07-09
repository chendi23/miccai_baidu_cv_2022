# -*- coding: utf-8 -*-
"""
@Time : 2022/6/26 1:16 下午
@Auth : zcd_zhendeshuai
@File : trainer.py
@IDE  : PyCharm

"""
import cv2
import keras.callbacks
import numpy as np
import skimage.io
import tensorflow as tf
from config import global_variables as gl
from utils.common_utils import parse_example, get_file_list, MyMIOU, dice_coef,dice_loss,dice_ce,dice_omit_ce,ce
from networks.hrnetV2 import hrnet_v2, hrnet18_v2, hrnet32_v2, hrnet48_v2
from networks.unet import NestedUNet, U_Net,AttU_Net

class SegNet:
    def __init__(self):
        self.n_class = gl.CLASSES
        self.loss_type = 'dice_ce'
        self.optimizer_name = 'adam'
        self.pretrained_ckpt_dir = ''
        self.batch_size = gl.BATCH_SIZE
        self.epoch = gl.EPOCH
        self.n_class = gl.CLASSES
        self.lr_warmup = gl.LR_WARMUP
        self.lr_start = gl.LR_START

    def prepare_input(self, tfrecord_dir, split='train'):
        dir = get_file_list(tfrecord_dir)
        ds = tf.data.TFRecordDataset(dir, buffer_size=self.batch_size * self.batch_size).map(
            lambda x: parse_example(x,split=split), num_parallel_calls=4).batch(
            self.batch_size, drop_remainder=False).prefetch(1)
        return ds

    def build_model(self):
        x = tf.keras.layers.Input(shape=gl.RESIZED_IMG)
        out = hrnet32_v2(x, n_class=self.n_class, include_top=True, mode="seg")  # mode = seg > hrnet v2 + semantic segmentation, clsf > hrnet v2 + classifier, ocr > hrnet v2 + ocr + semantic segmentation
        # print(out.shape)
        out = tf.keras.layers.UpSampling2D((4, 4))(out)
        model = tf.keras.Model(x, out)

        # model = AttU_Net(classes=gl.CLASSES)
        # # model.build((None, *gl.RESIZED_IMG))
        return model



    def get_loss(self):
        if self.loss_type == 'ce':
            loss = ce
        elif self.loss_type == 'dice':
            loss = dice_loss
        elif self.loss_type == 'dice_ce':
            loss = dice_ce
        else:
            loss = dice_omit_ce
        return loss

    def get_optimizer(self):
        ###TO DO: lr decay strategy ###
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr_start,
            decay_steps=self.epoch * gl.TOTAL_IMGS * gl.TRAIN_SET_RATIO // self.batch_size,
            decay_rate=0.9)

        if self.optimizer_name == 'adam':
            opt = tf.keras.optimizers.Adam(lr_schedule)
        else:
            opt = tf.keras.optimizers.SGD(lr_schedule, momentum=0.9, nesterov=True)
        return opt

    def fit(self):
        model = self.build_model()
        loss = self.get_loss()
        opt = self.get_optimizer()
        monitor=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='auto')
        saver = tf.keras.callbacks.ModelCheckpoint(filepath='./ckpt/', save_weights_only=True,save_best_only=True)
        model.compile(loss=loss, optimizer=opt, metrics=[dice_coef, MyMIOU(num_classes=4)])


        train_set = self.prepare_input(gl.TRAIN_DIR)
        val_set = self.prepare_input(gl.VAL_DIR)
        model.fit(x=train_set, y=None, epochs=self.epoch, validation_data=val_set,callbacks=[monitor,saver])

    def predict(self,load_from_weights=True, save_predictions=False):
        test_set = self.prepare_input(gl.TEST_DIR,split='test')
        if load_from_weights:
            model = self.build_model()
            model.compile()
            model.load_weights('./ckpt/dice/')
        else:
            dependencies = {
                'dice_coef': dice_coef,
                'my_miou':MyMIOU(num_classes=4)
            }

            model = tf.keras.models.load_model('./ckpt/',custom_objects=dependencies)
        out = model.predict(x=test_set,verbose=2)
        predict = np.argmax(out, axis=-1)
        predict[predict==0] = 255
        predict[predict==1] = 0
        predict[predict==2] = 80
        predict[predict==3] = 160
        predict = predict.astype(np.float32)

        if save_predictions:
            for i in range(predict.shape[0]):
                curr = cv2.resize(predict[i], gl.ORIGINAL_IMG_SIZE)
                # cv2.imwrite('/Users/chendi/data_competition/cv/miccai_2022_baidu/pre_result/Layer_Segmentations/0%d.png' % (predict.shape[0]+i+1), curr)
                cv2.imwrite('./res/0%d.png' % (predict.shape[0]+i+1), curr)



def main():
    net = SegNet()
    # net.fit()
    net.predict(load_from_weights=True,save_predictions=True)

if __name__ == '__main__':
    main()
