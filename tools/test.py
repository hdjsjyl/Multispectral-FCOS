# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
import tensorflow as tf
import time
import cv2
import argparse
import numpy as np
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network_v3 as build_whole_network
from libs.box_utils import draw_box_in_img
from help_utils import tools


def detect(det_net, inference_save_path, rgb_real_test_imgname_list, ir_real_test_imgname_list):

    # 1. preprocess img
    rgb_img_plac = tf.placeholder(tf.uint8, [None, None, 3], 'rgb')  # is RGB. not GBR
    rgb_img_batch = tf.cast(rgb_img_plac, tf.float32)
    rgb_img_batch = short_side_resize_for_inference_data(img_tensor=rgb_img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    ir_img_plac = tf.placeholder(tf.uint8, [None, None, 3], 'ir')  # is RGB. not GBR
    ir_img_batch = tf.cast(ir_img_plac, tf.float32)
    ir_img_batch = short_side_resize_for_inference_data(img_tensor=ir_img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    rgb_img_batch = rgb_img_batch - tf.constant(cfgs.RGB_PIXEL_MEAN)
    ir_img_batch = ir_img_batch - tf.constant(cfgs.IR_PIXEL_MEAN)
    #img_batch = (img_batch - tf.constant(cfgs.PIXEL_MEAN)) / (tf.constant(cfgs.PIXEL_STD)*255)
    rgb_img_batch = tf.expand_dims(rgb_img_batch, axis=0) # [1, None, None, 3]
    ir_img_batch = tf.expand_dims(ir_img_batch, axis=0) # [1, None, None, 3]

    detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
        rgb_input_img_batch=rgb_img_batch, ir_input_img_batch=ir_img_batch, seg_mask_batch = None,
        gtboxes_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt, model_variables = det_net.get_restorer_test()
    print(restore_ckpt)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        for i, rgb_img_name in enumerate(rgb_real_test_imgname_list):

            rgb_raw_img = cv2.imread(rgb_img_name)[:, :, ::-1]
            ir_raw_img = cv2.imread(ir_real_test_imgname_list[i])[:, :, ::-1]
            start = time.time()
            rgb_resized_img, ir_resized_img, detected_boxes, detected_scores, detected_categories = \
                sess.run(
                    [rgb_img_batch, ir_img_batch, detection_boxes, detection_scores, detection_category],
                    feed_dict={rgb_img_plac: rgb_raw_img, ir_img_plac: ir_raw_img}  
                )
            end = time.time()
            # print("{} cost time : {} ".format(img_name, (end - start)))

            raw_h, raw_w = ir_raw_img.shape[0], rgb_raw_img.shape[1]

            xmin, ymin, xmax, ymax = detected_boxes[:, 0], detected_boxes[:, 1], \
                                     detected_boxes[:, 2], detected_boxes[:, 3]

            resized_h, resized_w = rgb_resized_img.shape[1], ir_resized_img.shape[2]

            xmin = xmin * raw_w / resized_w
            xmax = xmax * raw_w / resized_w

            ymin = ymin * raw_h / resized_h
            ymax = ymax * raw_h / resized_h

            detected_boxes = np.transpose(np.stack([xmin, ymin, xmax, ymax]))

            show_indices = detected_scores >= cfgs.SHOW_SCORE_THRSHOLD
            show_scores = detected_scores[show_indices]
            show_boxes = detected_boxes[show_indices]
            show_categories = detected_categories[show_indices]

            nake_name = rgb_img_name.split('/')[-1]
            f1  = open(inference_save_path + '/txt/' + nake_name.split('.')[0]+'.txt', 'w')
            final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(rgb_raw_img - np.array(cfgs.RGB_PIXEL_MEAN),
                                                                                boxes=show_boxes,
                                                                                labels=show_categories,
                                                                                scores=show_scores, txt_file=f1, 
                                                                                img_name=nake_name.split('.')[0])
            # print (inference_save_path + '/' + nake_name)
            cv2.imwrite(inference_save_path + '/img/' + nake_name,
                        final_detections[:, :, ::-1])

            tools.view_bar('{} image cost {}s'.format(rgb_img_name, (end - start)), i + 1, len(ir_real_test_imgname_list))


def test(rgb_test_dir, ir_test_dir, inference_save_path):

    rgb_test_imgname_list = [os.path.join(rgb_test_dir, img_name) for img_name in sorted(os.listdir(rgb_test_dir))
                                                          if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    ir_test_imgname_list = [os.path.join(ir_test_dir, img_name) for img_name in sorted(os.listdir(ir_test_dir))
                                                          if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    assert len(rgb_test_imgname_list) != 0, 'test_dir has no imgs there.' \
                                        ' Note that, we only support img format of (.jpg, .png, and .tiff) '
    assert len(ir_test_imgname_list) != 0, 'test_dir has no imgs there.' \
                                        ' Note that, we only support img format of (.jpg, .png, and .tiff) '

    faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False, batch_size=1)
    detect(det_net=faster_rcnn, inference_save_path=inference_save_path, rgb_real_test_imgname_list=rgb_test_imgname_list, ir_real_test_imgname_list=ir_test_imgname_list)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='TestImgs...U need provide the test dir')
    parser.add_argument('--rgb_data_dir', dest='rgb_data_dir',
                        help='rgb_data path',
                        default='demos', type=str)
    parser.add_argument('--ir_data_dir', dest='ir_data_dir',
                        help='ir_data path',
                        default='demos', type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='demo imgs to save',
                        default='inference_results', type=str)
    parser.add_argument('--GPU', dest='GPU',
                        help='gpu id ',
                        default='0', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    test(args.rgb_data_dir, args.ir_data_dir,
         inference_save_path=args.save_dir)
















