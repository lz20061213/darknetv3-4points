#!/usr/bin/env python
# encoding: utf-8

# --------------------------------------------------------
# file: tensorflow_darknet.py
# Copyright(c) 2017-2020 SeetaTech
# Written by Zhuang Liu
# 10:20
# --------------------------------------------------------

import tensorflow as tf
import os
import io
from cStringIO import StringIO
# from io import BytesIO as StringIO
from collections import defaultdict
import configparser
import numpy as np
import cv2
import sys

# reload(sys)
# sys.setdefaultencoding('utf-8')

print('Using Tensorflow ' + tf.__version__)

gpu_device = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True


def main():
    cfg_path = '../cfg/yolov3.cfg'
    weights_path = '../yolov3.weights'
    classname_path = '../data/coco.names'
    im_path = '../data/giraffe.jpg'
    input_dim = [416, 416, 3]
    classnames = get_classes(classname_path)
    nms = True

    im = cv2.imread(im_path)
    sized = letter_box(im, input_dim)
    input_img, out_layers, anchors, masks, check_tensors, layer_cats = yolov3_architecture(cfg_path, weights_path)

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        outputs, check_outputs = sess.run([out_layers, check_tensors], feed_dict={input_img: sized})
        # for i in range(len(check_outputs)):
        #     print layer_cats[i], check_outputs[i].shape
        #     if layer_cats[i] == 'input':
        #         print check_outputs[i][0, 200:205, 200:205, 0]
        #     elif layer_cats[i] == 'upsample':
        #         print check_outputs[i][0, 1, 0, 0:10]
        #     elif layer_cats[i] == 'route':
        #         if check_outputs[i].shape[3] > 256:
        #             print check_outputs[i][0, 0, 0, 256:266]
        #         else:
        #             print check_outputs[i][0, 0, 0, 0:10]
        #     else:
        #         print check_outputs[i][0, 0, 0, 0:10]
        dets = get_yolov3_detections(anchors, masks, outputs, im.shape[1], im.shape[0], classnum=len(classnames))
        if nms:
            dets = do_nms_sort(dets, len(classnames), thresh=0.45)

    draw_detection_results(im, dets, classnames)


def get_classes(path):
    classes = []
    names = open(path, 'r').readlines()
    for name in names:
        classes.append(name.strip())
    return classes


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.
    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def Convolutional(X, conv_weights, stride, pad, conv_bias, bn_weights_list, batch_normalize, activation, scope=None,
                  reuse=False):
    with tf.variable_scope(scope or 'convoluational', reuse=reuse):
        conv_weights = tf.get_variable('conv_weights', conv_weights.shape, trainable=False,
                                       initializer=tf.constant_initializer(conv_weights))
        if pad != 0:
            X = tf.pad(X, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'CONSTANT')
        h = tf.nn.conv2d(X, conv_weights, strides=[1, stride, stride, 1], padding='VALID')  # pad before use conv2d
        if batch_normalize:
            bn_gamma = bn_weights_list[0]
            bn_beta = bn_weights_list[1]
            bn_rolling_mean = bn_weights_list[2]
            bn_rolling_variance = bn_weights_list[3]
            h = tf.layers.batch_normalization(h, epsilon=1e-5, beta_initializer=tf.constant_initializer(bn_beta),
                                              gamma_initializer=tf.constant_initializer(bn_gamma),
                                              moving_mean_initializer=tf.constant_initializer(bn_rolling_mean),
                                              moving_variance_initializer=tf.constant_initializer(bn_rolling_variance),
                                              training=False, trainable=False)
        else:
            h += conv_bias

        assert activation == 'leaky' or activation == 'linear', 'activation should be either leaky or linear.'
        if activation == 'leaky':
            h = tf.nn.leaky_relu(h, alpha=0.1)
        elif activation == 'linear':
            pass
    return h


def MaxPooling(X, size, stride, pad, scope=None, reuse=False):
    with tf.variable_scope(scope or 'maxpool', reuse=reuse):
        h = tf.pad(X, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'CONSTANT')
        h = tf.nn.max_pool(h, [1, size, size, 1], stirdes=[1, stride, stride, 1],
                           padding='VALID', name=None)
    return h


def Upsample(X, stride, scope=None, reuse=False):
    with tf.variable_scope(scope or 'upsample', reuse=reuse):
        input_shape = tf.shape(X)
        h = tf.image.resize_images(X, size=[input_shape[1] * stride, input_shape[2] * stride],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return h


def parse_anchor(anchors_string):
    anchors = []
    lengths = [int(i) for i in anchors_string.split(',')]
    for i in range(len(lengths) / 2):
        anchors.append([lengths[2 * i], lengths[2 * i + 1]])
    return anchors


def yolov3_architecture(cfg_path, weights_path):
    # Load weights and config
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(
        shape=(3,), dtype='int32', buffer=weights_file.read(12)
    )
    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    # Parsing Darknet config
    print('Parsing Darknet config.')
    unique_config_file = unique_config_sections(cfg_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    # Creating tensorflow model
    print('Creating tensorflow model')
    input_img = tf.placeholder(tf.float32, [None, None, 3], name='input_img')
    input_layer = tf.expand_dims(input_img, axis=0)
    all_layers = []

    all_layers.append(input_layer)
    prev_layer = input_layer

    count = 0
    out_layers = []
    anchors = None
    masks = []
    layer_cats = []
    check_tensors = []

    check_tensors.append(tf.identity(input_layer))
    layer_cats.append('input')

    for section in cfg_parser._sections:
        print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(size / 2)
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            # Setting weights
            # Darknet serializes convolutional weights as [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = prev_layer.get_shape().as_list()
            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_weights_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn' if batch_normalize else ' ', activation, weights_shape)

            conv_bias = np.ndarray(
                shape=(filters,),
                dtype='float32',
                buffer=weights_file.read(filters * 4)
            )
            count += filters

            bn_weight_list = None
            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12)
                )
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],
                    conv_bias,
                    bn_weights[1],
                    bn_weights[2]
                ]

            conv_weights = np.ndarray(
                shape=darknet_weights_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4)
            )

            count += weights_size

            # darknet conv_weights asr serialized caffe-style: (out_dim, in_dim, height, width)
            # we would like to set these to  tensorflow order: (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

            # Create convolutional layer
            conv_layer = Convolutional(prev_layer, conv_weights, stride, pad, conv_bias, bn_weight_list,
                                       batch_normalize, activation, scope=section, reuse=False)

            check_tensors.append(tf.identity(conv_layer))
            layer_cats.append('conv')

            all_layers.append(conv_layer)
            prev_layer = conv_layer

        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] if i < 0 else all_layers[i + 1] for i in ids]
            if len(layers) > 1:
                print('Concatenating rote layers:', layers)
                concatenate_layer = tf.concat(layers, axis=-1)
                check_tensors.append(tf.identity(concatenate_layer))
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                skip_layer = layers[0]
                check_tensors.append(tf.identity(skip_layer))
                all_layers.append(skip_layer)
                prev_layer = skip_layer

            layer_cats.append('route')

        elif section.startswith('maxpool'):  # no this layer in yolov3.cfg
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int((size - 1) / 2)
            maxpool_layer = MaxPooling(prev_layer, size, stride, pad, scope=section, reuse=False)
            all_layers.append(maxpool_layer)
            prev_layer = maxpool_layer

        elif section.startswith('shortcut'):
            index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            assert activation == 'linear', 'Only linear activation supported.'
            shortcut_layer = all_layers[index] + prev_layer
            check_tensors.append(tf.identity(shortcut_layer))
            all_layers.append(shortcut_layer)
            prev_layer = shortcut_layer

            layer_cats.append('shortcut')

        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            assert stride == 2, 'Only stride=2 supported.'
            upsample_layer = Upsample(prev_layer, stride, scope=section, reuse=False)
            check_tensors.append(tf.identity(upsample_layer))
            all_layers.append(upsample_layer)
            layer_cats.append('upsample')
            prev_layer = upsample_layer

        elif section.startswith('yolo'):
            if anchors is None:
                anchors = cfg_parser[section]['anchors']
                anchors = parse_anchor(anchors)
            mask = [int(i) for i in cfg_parser[section]['mask'].split(',')]
            masks.append(mask)
            out_layers.append(all_layers[-1])
            check_tensors.append(tf.identity(all_layers[-1]))
            layer_cats.append('yolo')
            all_layers.append(None)
            prev_layer = all_layers[-1]

        elif section.startswith('net'):
            pass

        else:
            raise ValueError('Unsupported section header type: {}'.format(section))

    return input_img, out_layers, anchors, masks, check_tensors, layer_cats


def resize_image(im, (w, h)):
    def get_pixel(im, x, y, c):
        # print x, y, c
        assert (x < im.shape[1] and y < im.shape[0] and c < im.shape[2])
        return im[y, x, c]

    def set_pixel(im, x, y, c, val):
        if (x < 0 or y < 0 or c < 0 or x >= im.shape[1] or y >= im.shape[0] or c >= im.shape[2]):
            return
        assert (x < im.shape[1] and y < im.shape[0] and c < im.shape[2])
        im[y, x, c] = val

    def add_pixel(im, x, y, c, val):
        assert (x < im.shape[1] and y < im.shape[0] and c < im.shape[2])
        im[y, x, c] += val

    sized = np.zeros(shape=(h, w, im.shape[2]), dtype=np.float32)
    w_scale = float(im.shape[1] - 1) / (w - 1)
    h_scale = float(im.shape[0] - 1) / (h - 1)

    # print im.shape
    for k in range(im.shape[2]):
        for r in range(im.shape[0]):
            for c in range(w):
                # print c, r, k
                if (c == w - 1 or im.shape[1] == 1):
                    val = get_pixel(im, im.shape[1] - 1, r, k)
                else:
                    sx = c * w_scale
                    ix = int(sx)
                    dx = sx - ix
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k)
                set_pixel(im, c, r, k, val)

    for k in range(im.shape[2]):
        for r in range(h):
            sy = r * h_scale
            iy = int(sy)
            dy = sy - iy
            for c in range(w):
                val = (1 - dy) * get_pixel(im, c, iy, k)
                set_pixel(sized, c, r, k, val)
            if (r == h - 1 or im.shape[0] == 1): continue
            for c in range(w):
                val = dy * get_pixel(im, c, iy + 1, k)
                add_pixel(sized, c, r, k, val)

    return sized


def letter_box(im, input_dim=[416, 416, 3]):  # [h, w]

    im_copy = im.copy()

    swap = im_copy[:, :, 0].copy()
    im_copy[:, :, 0] = im_copy[:, :, 2]
    im_copy[:, :, 2] = swap

    if float(input_dim[1]) / im_copy.shape[1] < float(input_dim[0]) / im_copy.shape[0]:
        new_w = input_dim[1]
        new_h = im_copy.shape[0] * input_dim[1] / im_copy.shape[1]
    else:
        new_h = input_dim[0]
        new_w = im_copy.shape[1] * input_dim[0] / im_copy.shape[0]

    sized = cv2.resize(im_copy, (new_w, new_h), None, 0.0, 0.0, cv2.INTER_LINEAR)
    sized = np.float32(sized) / 255.0

    boxed = np.ones(shape=input_dim, dtype=np.float32) * 0.5
    fill_loc_y = int((input_dim[1] - new_h) / 2)
    fill_loc_x = int((input_dim[0] - new_w) / 2)
    boxed[fill_loc_y:fill_loc_y + new_h, fill_loc_x:fill_loc_x + new_w, :] = sized

    return boxed


def letter_box_ori(im, input_dim=[416, 416, 3]):  # [h, w]

    im = im / 255.0

    # print 'ori im after cv.read():'
    # print im[200:205, 200:205, 0]

    swap = im[:, :, 0].copy()
    im[:, :, 0] = im[:, :, 2]
    im[:, :, 2] = swap

    # print 'ori im:'
    # print im[200:205, 200:205, 0]

    if float(input_dim[1]) / im.shape[1] < float(input_dim[0]) / im.shape[0]:
        new_w = input_dim[1]
        new_h = im.shape[0] * input_dim[1] / im.shape[1]
    else:
        new_h = input_dim[0]
        new_w = im.shape[1] * input_dim[0] / im.shape[0]

    sized = resize_image(im, (new_w, new_h))
    # print 'sized im:'
    # print sized[200:205, 200:205, 0]
    # sized = np.float32(sized) / 255.0
    boxed = np.ones(shape=input_dim, dtype=np.float32) * 0.5
    fill_loc_y = int((input_dim[1] - new_h) / 2)
    fill_loc_x = int((input_dim[0] - new_w) / 2)
    boxed[fill_loc_y:fill_loc_y + new_h, fill_loc_x:fill_loc_x + new_w, :] = sized

    return boxed


def logistic_activate(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_yolo_box(bbox, anchor, i, j, lw, lh, w, h):
    bx = (i + logistic_activate(bbox[0])) / lw
    by = (j + logistic_activate(bbox[1])) / lh
    bw = np.exp(bbox[2]) * anchor[0] / w
    bh = np.exp(bbox[3]) * anchor[1] / h
    return [bx, by, bw, bh]


def correct_yolo_boxes(dets, w, h, input_dim=[416, 416, 3]):  # (h, w, c)
    if float(input_dim[1]) / w < float(input_dim[0]) / h:
        new_w = input_dim[1]
        new_h = (h * input_dim[1]) / w
    else:
        new_h = input_dim[0]
        new_w = (w * input_dim[0]) / h

    for i in range(len(dets)):
        bbox = dets[i]['bbox']
        bbox[0] = (bbox[0] - (input_dim[1] - new_w) / 2.0 / input_dim[1]) / (float(new_w) / input_dim[1])
        bbox[1] = (bbox[1] - (input_dim[0] - new_h) / 2.0 / input_dim[0]) / (float(new_h) / input_dim[0])
        bbox[2] = bbox[2] * (float(input_dim[1]) / new_w)
        bbox[3] = bbox[3] * (float(input_dim[0]) / new_h)
        dets[i]['bbox'] = bbox

    return dets


def get_yolov3_detections(anchors, masks, outputs, w, h, thresh=0.5, classnum=1, input_dim=[416, 416, 3]):
    assert len(masks) == len(outputs), 'length of masks should be same with outputs.'
    dets = []
    count = 0
    for i in range(len(masks)):
        mask = masks[i]
        output = outputs[i][0]
        # get results, first do activate of the different outputs
        for row in range(output.shape[0]):
            for col in range(output.shape[1]):
                for n in range(len(mask)):
                    objectness = logistic_activate(output[row, col, n * (4 + 1 + classnum) + 4])
                    if objectness <= thresh: continue
                    # print("objectness:%d %d %d %f\n", row, col, n, objectness)
                    det = {}
                    det['bbox'] = get_yolo_box(
                        output[row, col, n * (4 + 1 + classnum):n * (4 + 1 + classnum) + 4], anchors[mask[n]], col, row,
                        output.shape[1], output.shape[0], input_dim[1], input_dim[0])
                    # print("bbox: %f %f %f %f\n", det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3])
                    det['objectness'] = objectness
                    det['classes'] = classnum
                    det['prob'] = []
                    for c in range(classnum):
                        prob = objectness * logistic_activate(output[row, col, n * (4 + 1 + classnum) + 4 + 1 + c])
                        if prob > thresh:
                            det['prob'].append(prob)
                        else:
                            det['prob'].append(0)
                    dets.append(det)
                    count += 1
    # correct_yolo_boxes
    dets = correct_yolo_boxes(dets, w, h, input_dim)
    return dets


def box_iou(bboxA, bboxB):
    def overlap(x1, w1, x2, w2):
        l1 = x1 - w1 / 2
        l2 = x2 - w2 / 2
        if l1 > l2:
            left = l1
        else:
            left = l2
        r1 = x1 + w1 / 2
        r2 = x2 + w2 / 2
        if r1 < r2:
            right = r1
        else:
            right = r2
        return right - left

    def box_intersection(bboxA, bboxB):
        w = overlap(bboxA[0], bboxA[2], bboxB[0], bboxB[2])
        h = overlap(bboxA[1], bboxA[3], bboxB[1], bboxB[3])
        if w < 0 or h < 0:
            return 0
        return w * h

    def box_union(bboxA, bboxB):
        i = box_intersection(bboxA, bboxB)
        u = bboxA[2] * bboxA[3] + bboxB[2] * bboxA[3] - i
        return u

    return box_intersection(bboxA, bboxB) / box_union(bboxA, bboxB)


def num_comparator(bboxA, bboxB):
    if bboxB['sort_class'] >= 0:
        diff = bboxA['prob'][bboxB['sort_class']] - bboxB['prob'][bboxB['sort_class']]
    else:
        diff = bboxA['objectness'] - bboxB['objectness']
    if diff < 0:
        return 1
    elif diff > 0:
        return -1
    return 0


# no use in test_forward
def do_nms_obj(dets, classnum=1, thresh=0.4):
    new_dets = []
    for i in range(len(dets)):
        if dets[i]['objectness'] != 0:
            new_dets.append(dets[i])

    for i in range(len(new_dets)):
        new_dets[i]['sort_class'] = -1

    new_dets = sorted(new_dets, cmp=num_comparator, reverse=True)

    for i in range(len(new_dets)):
        if new_dets[i]['objectness'] == 0: continue
        bboxA = dets[i]['bbox']
        for j in range(i + 1, len(new_dets)):
            if dets[j]['objectness'] == 0: continue
            bboxB = dets[j]['bbox']
            if box_iou(bboxA, bboxB) > thresh:
                dets[j]['objectness'] = 0
                for c in range(classnum):
                    dets[j]['prob'][c] = 0

    return new_dets


def do_nms_sort(dets, classnum=1, thresh=0.4):
    new_dets = []
    for i in range(len(dets)):
        if dets[i]['objectness'] != 0:
            new_dets.append(dets[i])

    for c in range(classnum):
        for i in range(len(new_dets)):
            new_dets[i]['sort_class'] = c
        new_dets = sorted(new_dets, cmp=num_comparator)
        # print "Begin"
        # for i in range(len(new_dets)):
        #     print(new_dets[i]['objectness'])
        # print "After"
        for i in range(len(new_dets)):
            if new_dets[i]['prob'][c] == 0: continue
            bboxA = new_dets[i]['bbox']
            for j in range(i + 1, len(new_dets)):
                bboxB = new_dets[j]['bbox']
                if box_iou(bboxA, bboxB) > thresh:
                    new_dets[j]['prob'][c] = 0

    return new_dets


colors = [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]


def get_color(x, num):
    ratio = float(x) / num * 5
    i = int(np.floor(ratio))
    j = int(np.ceil(ratio))
    ratio -= i
    r = (1 - ratio) * colors[i][2] + ratio * colors[j][2]
    g = (1 - ratio) * colors[i][1] + ratio * colors[j][1]
    b = (1 - ratio) * colors[i][0] + ratio * colors[j][0]
    color = (int(b * 255), int(g * 255), int(r * 255))
    return color


def draw_detection_results(im, dets, classnames, thresh=0.5):
    im_copy = im.copy()
    for det in dets:
        class_ind = -1
        for j in range(len(classnames)):
            if det['prob'][j] > thresh:
                if class_ind < 0:
                    class_ind = j
        if class_ind >= 0:
            classname = classnames[class_ind]
            offset = class_ind * 123457 % len(classnames)
            color = get_color(offset, len(classnames))
            bbox = det['bbox']
            left = int((bbox[0] - bbox[2] / 2.0) * im.shape[1])
            right = int((bbox[0] + bbox[2] / 2.0) * im.shape[1])
            top = int((bbox[1] - bbox[3] / 2.0) * im.shape[0])
            bottom = int((bbox[1] + bbox[3] / 2.0) * im.shape[0])

            if left < 0: left = 0
            if right > im.shape[1] - 1: right = im.shape[1] - 1
            if top < 0: top = 0
            if bottom > im.shape[0] - 1: bottom = im.shape[0] - 1

            cv2.rectangle(im_copy, (left, top), (right, bottom), color, 2, 8)

            text = '{}: {}'.format(classname, round(det['prob'][class_ind], 4))
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            text_size = cv2.getTextSize(text, font_face, font_scale, thickness)

            cv2.rectangle(im_copy, (left - 1, top - 20), (left + text_size[0][0] + 10, top), color, -1)

            cv2.putText(im_copy, text, (left + 5, top - 5), font_face, font_scale, (0, 0, 0))

    cv2.imshow('result', im_copy)
    cv2.waitKey(0)


if __name__ == '__main__':
    sys.exit(main())
