# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division


import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
# import tfplot as tfp


def resnet_arg_scope(
        is_training=True, weight_decay=cfgs.WEIGHT_DECAY, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def fusion_two_layer(C_i, P_j, scope):
    '''
    i = j+1
    :param C_i: shape is [1, h, w, c]
    :param P_j: shape is [1, h/2, w/2, 256]
    :return:
    P_i
    '''
    with tf.variable_scope(scope):
        level_name = scope.split('_')[1]

        h, w = tf.shape(C_i)[1], tf.shape(C_i)[2]
        upsample_p = tf.image.resize_bilinear(P_j,
                                              size=[h, w],
                                              name='up_sample_'+level_name)

        reduce_dim_c = slim.conv2d(C_i,
                                   num_outputs=256,
                                   kernel_size=[1, 1], stride=1,
                                   scope='reduce_dim_'+level_name)

        add_f = 0.5*upsample_p + 0.5*reduce_dim_c

        # P_i = slim.conv2d(add_f,
        #                   num_outputs=256, kernel_size=[3, 3], stride=1,
        #                   padding='SAME',
        #                   scope='fusion_'+level_name)
        return add_f


# def add_heatmap(feature_maps, name):
#     '''
#
#     :param feature_maps:[B, H, W, C]
#     :return:
#     '''
#
#     def figure_attention(activation):
#         fig, ax = tfp.subplots()
#         im = ax.imshow(activation, cmap='jet')
#         fig.colorbar(im)
#         return fig
#
#     heatmap = tf.reduce_sum(feature_maps, axis=-1)
#     heatmap = tf.squeeze(heatmap, axis=0)
#     tfp.summary.plot(name, figure_attention, [heatmap])


def resnet_base(rgb_img_batch, ir_img_batch, scope_name, is_training=True):

    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. ')
    org_scope_name = scope_name
    blocks = [resnet_v1_block('RGB/resnet_v1_50/block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('RGB/resnet_v1_50/block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_block('RGB/resnet_v1_50/block3', base_depth=256, num_units=middle_num_units, stride=2),
              resnet_v1_block('RGB/resnet_v1_50/block4', base_depth=512, num_units=3, stride=1)]
    # when use fpn . stride list is [1, 2, 2]

    scope_name = "RGB/"+org_scope_name

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net_rgb = resnet_utils.conv2d_same(
                rgb_img_batch, 64, 7, stride=2, scope='conv1')
            net_rgb = tf.pad(net_rgb, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net_rgb = slim.max_pool2d(
                net_rgb, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2_rgb, end_points_C2_rgb = resnet_v1.resnet_v1(net_rgb,
                                                blocks[0:1],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')
    # add_heatmap(C2, name='Layer2/C2_heat')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3_rgb, end_points_C3_rgb = resnet_v1.resnet_v1(C2_rgb,
                                                blocks[1:2],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')
    C3_rgb = slim.repeat(C3_rgb, 1, slim.conv2d, 256, [1, 1], activation_fn = None, scope = 'conv_resize_rgb')
    # add_heatmap(C3, name='Layer3/C3_heat')

    blocks = [resnet_v1_block('IR/resnet_v1_50/block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('IR/resnet_v1_50/block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_block('IR/resnet_v1_50/block3', base_depth=256, num_units=middle_num_units, stride=2),
              resnet_v1_block('IR/resnet_v1_50/block4', base_depth=512, num_units=3, stride=1)]

    scope_name = "IR/"+org_scope_name

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net_ir = resnet_utils.conv2d_same(
                ir_img_batch, 64, 7, stride=2, scope='conv1')
            net_ir = tf.pad(net_ir, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net_ir = slim.max_pool2d(
                net_ir, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2_ir, end_points_C2_ir = resnet_v1.resnet_v1(net_ir,
                                                blocks[0:1],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')
    # add_heatmap(C2, name='Layer2/C2_heat')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3_ir, end_points_C3_ir = resnet_v1.resnet_v1(C2_ir,
                                                blocks[1:2],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')
    C3_ir = slim.repeat(C3_ir, 1, slim.conv2d, 256, [1, 1],activation_fn = None, scope = 'conv_resize_ir')
    # add_heatmap(C3, name='Layer3/C3_heat')

    blocks = [resnet_v1_block('MULTI/resnet_v1_50/block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('MULTI/resnet_v1_50/block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_block('MULTI/resnet_v1_50/block3', base_depth=256, num_units=middle_num_units, stride=2),
              resnet_v1_block('MULTI/resnet_v1_50/block4', base_depth=512, num_units=3, stride=1)]

    scope_name = "MULTI/"+org_scope_name
   
    C3_multi = tf.concat(axis = 3, values = [C3_rgb, C3_ir])

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4_multi, end_points_C4_multi = resnet_v1.resnet_v1(C3_multi,
                                                blocks[2:3],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # add_heatmap(C4, name='Layer4/C4_heat')

    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5_multi, end_points_C5_multi = resnet_v1.resnet_v1(C4_multi,
                                                blocks[3:4],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)
    # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')
    # add_heatmap(C5, name='Layer5/C5_heat')

    multi_end_points_C2 = tf.concat(axis=3, values = [end_points_C2_rgb['{}/block1/unit_2/bottleneck_v1'.format("RGB/resnet_v1_50/RGB/"+org_scope_name)], end_points_C2_ir['{}/block1/unit_2/bottleneck_v1'.format("IR/resnet_v1_50/IR/"+org_scope_name)]])

    multi_end_points_C3 = tf.concat(axis=3, values = [end_points_C3_rgb['{}/block2/unit_3/bottleneck_v1'.format("RGB/resnet_v1_50/RGB/"+org_scope_name)], end_points_C3_ir['{}/block2/unit_3/bottleneck_v1'.format("IR/resnet_v1_50/IR/"+org_scope_name)]])

    multi_end_points_C4 = end_points_C4_multi['{}/block3/unit_{}/bottleneck_v1'.format("MULTI/resnet_v1_50/MULTI/"+org_scope_name, middle_num_units - 1)]

    multi_end_points_C5 = end_points_C5_multi['{}/block4/unit_3/bottleneck_v1'.format("MULTI/resnet_v1_50/MULTI/"+org_scope_name)]

    feature_dict = {'C2': multi_end_points_C2,
                    'C3': multi_end_points_C3,
                    'C4': multi_end_points_C4,
                    'C5': multi_end_points_C5,
                    # 'C5': end_points_C5['{}/block4'.format(scope_name)],
                    }

    scope_name = org_scope_name

    pyramid_dict = {}
    with tf.variable_scope('build_pyramid'):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY),
                            activation_fn=None, normalizer_fn=None):

            conv_channels = 256
            last_fm = None
            for i in range(3):
                fm = feature_dict['C{}'.format(5-i)]
                fm_1x1_conv = slim.conv2d(fm,  num_outputs=conv_channels, kernel_size=[1, 1],
                                          stride=1, scope='p{}_1x1_conv'.format(5-i))
                if last_fm is not None:
                    h, w = tf.shape(fm_1x1_conv)[1], tf.shape(fm_1x1_conv)[2]
                    last_resize = tf.image.resize_bilinear(last_fm,
                                                           size=[h, w],
                                                           name='p{}_up2x'.format(5-i))

                    fm_1x1_conv = fm_1x1_conv + last_resize

                last_fm = fm_1x1_conv

                fm_3x3_conv = slim.conv2d(fm_1x1_conv,
                                          num_outputs=conv_channels, kernel_size=[3, 3], padding="SAME",
                                          stride=1, scope='p{}_3x3_conv'.format(5 - i))
                pyramid_dict['P{}'.format(5-i)] = fm_3x3_conv

            p6 = slim.conv2d(pyramid_dict['P5'],
                             num_outputs=conv_channels, kernel_size=[3, 3], padding="SAME",
                             stride=2, scope='p6_conv')
            pyramid_dict['P6'] = p6

            p7 = tf.nn.relu(p6)

            p7 = slim.conv2d(p7,
                             num_outputs=conv_channels, kernel_size=[3, 3], padding="SAME",
                             stride=2, scope='p7_conv')

            pyramid_dict['P7'] = p7

    # for level in range(7, 1, -1):
    #     add_heatmap(pyramid_dict['P%d' % level], name='Layer%d/P%d_heat' % (level, level))

    return pyramid_dict































