# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import os
from data.io import image_preprocess_multi_gpu_v2 as image_preprocess
from libs.configs import cfgs

def read_single_example_and_decode(filename_queue):

    # tfrecord_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    # reader = tf.TFRecordReader(options=tfrecord_options)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'rgb_img': tf.FixedLenFeature([], tf.string),
            'ir_img': tf.FixedLenFeature([], tf.string),
            'seg_mask': tf.FixedLenFeature([], tf.string),
            'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
            'num_objects': tf.FixedLenFeature([], tf.int64)
        }
    )
    img_name = features['img_name']
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)
    rgb_img = tf.decode_raw(features['rgb_img'], tf.uint8)
    ir_img = tf.decode_raw(features['ir_img'], tf.uint8)
    seg_mask = tf.decode_raw(features['seg_mask'], tf.uint8)

    rgb_img = tf.reshape(rgb_img, shape=[img_height, img_width, 3])
    ir_img = tf.reshape(ir_img, shape=[img_height, img_width, 3])
    seg_mask = tf.reshape(seg_mask, shape=[img_height, img_width, 1])

    gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])

    num_objects = tf.cast(features['num_objects'], tf.int32)
    return img_name, rgb_img, ir_img, seg_mask, gtboxes_and_label, num_objects


def read_and_prepocess_single_img(filename_queue, shortside_len, is_training):

    img_name, rgb_img, ir_img, seg_mask, gtboxes_and_label, num_objects = read_single_example_and_decode(filename_queue)

    rgb_img = tf.cast(rgb_img, tf.float32)
    ir_img = tf.cast(ir_img, tf.float32)
    seg_mask = tf.cast(seg_mask, tf.float32)

    if is_training:
        rgb_img, gtboxes_and_label, img_h, img_w = image_preprocess.short_side_resize(img_tensor=rgb_img, gtboxes_and_label=gtboxes_and_label,
                                                                    target_shortside_len=shortside_len,
                                                                    length_limitation=cfgs.IMG_MAX_LENGTH)
        ir_img, _, _, _ = image_preprocess.short_side_resize(img_tensor=ir_img, gtboxes_and_label=gtboxes_and_label,
                                                                    target_shortside_len=shortside_len,
                                                                    length_limitation=cfgs.IMG_MAX_LENGTH)
       
        rgb_img, ir_img, seg_mask, gtboxes_and_label = image_preprocess.multi_random_flip_left_right(rgb_img_tensor=rgb_img, ir_img_tensor=ir_img, 	seg_mask_tensor = seg_mask, gtboxes_and_label = gtboxes_and_label)

    else:
         rgb_img, gtboxes_and_label, img_h, img_w = image_preprocess.short_side_resize(img_tensor=rgb_img, gtboxes_and_label=gtboxes_and_label,
                                                                                  target_shortside_len=shortside_len,
                                                                                  length_limitation=cfgs.IMG_MAX_LENGTH)
         ir_img, _ = image_preprocess.short_side_resize(img_tensor=ir_img, gtboxes_and_label=gtboxes_and_label,
                                                                                  target_shortside_len=shortside_len,
                                                                                  length_limitation=cfgs.IMG_MAX_LENGTH)

    if cfgs.NET_NAME in ['resnet101_v1d', 'resnet50_v1d']:
        rgb_img = rgb_img / 255 - tf.constant([[cfgs.PIXEL_MEAN_]])
        ir_img = ir_img / 255
        seg_mask = seg_mask / 255
    else:
        rgb_img = rgb_img - tf.constant([[cfgs.RGB_PIXEL_MEAN]])  # sub pixel mean at last
        ir_img = ir_img - tf.constant([[cfgs.IR_PIXEL_MEAN]])  # sub pixel mean at last
        seg_mask = seg_mask / 255

    return img_name, rgb_img, ir_img, seg_mask, gtboxes_and_label, num_objects, img_h, img_w


def next_batch(dataset_name, batch_size, shortside_len, is_training):
    '''
    :return:
    img_name_batch: shape(1, 1)
    img_batch: shape:(1, new_imgH, new_imgW, C)
    gtboxes_and_label_batch: shape(1, Num_Of_objects, 5] .each row is [x1, y1, x2, y2, label]
    '''
    # assert batch_size == 1, "we only support batch_size is 1.We may support large batch_size in the future"

    if dataset_name not in ['ship', 'spacenet', 'pascal', 'coco', 'bdd100k', 'DOTA', 'kaist']:
        raise ValueError('dataSet name must be in pascal, coco spacenet and ship')

    if is_training:
        pattern = os.path.join('../data/tfrecord', dataset_name + '_train*')
    else:
        pattern = os.path.join('../data/tfrecord', dataset_name + '_test*')

    print('tfrecord path is -->', os.path.abspath(pattern))

    filename_tensorlist = tf.train.match_filenames_once(pattern)

    filename_queue = tf.train.string_input_producer(filename_tensorlist)

    img_name, rgb_img, ir_img, seg_mask, gtboxes_and_label, num_obs, img_h, img_w = read_and_prepocess_single_img(filename_queue, shortside_len,
                                                                              is_training=is_training)
    img_name_batch, rgb_img_batch, ir_img_batch, seg_mask_batch, gtboxes_and_label_batch, num_obs_batch, img_h_batch, img_w_batch = \
        tf.train.batch(
                       [img_name, rgb_img, ir_img, seg_mask, gtboxes_and_label, num_obs, img_h, img_w],
                       batch_size=batch_size,
                       capacity=16,
                       num_threads=16,
                       dynamic_pad=True)
    return img_name_batch, rgb_img_batch, gtboxes_and_label_batch, num_obs_batch, img_h_batch, img_w_batch, ir_img_batch, seg_mask_batch
