#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : utils.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:14:19
#   Description :
#
# ================================================================

import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from core.config import cfg


def read_class_names(class_file_name):
    """
    loads class name from a file
    加载id和类别名称映射关系，文件中是每行一个类别名称
    :param class_file_name:
    :return:
    """
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    """
    loads the anchors from a file
    加载Anchor Box尺度信息
    :param anchors_path:
    :return:
    """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, -1, 2)


def image_preporcess(image, target_size, gt_boxes=None):
    """
    图像预处理相关操作(Resize&Padding操作)
    :param image:
    :param target_size:
    :param gt_boxes:
    :return:
    """
    # 1. 图像转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    # 2. 获取图像的原始大小以及目标大小
    ih, iw = target_size
    h, w, _ = image.shape

    # 3. 计算缩放比例以及按照该比例缩放后的图像大小
    # 按照宽度、高度的最小比例缩放
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)

    # 4. 图像大小缩放
    image_resized = cv2.resize(image, (nw, nh))

    # 5. 图像填充(因为是按照最小比例缩放的，所有某个方向是大小不达标的)
    # a. 构建一个目标大小的图像(灰度)
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    # b. 计算宽度和高度上需要填充的像素点数目(分为两部分)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    # c. 数据赋值
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    # d. 范围转换
    image_paded = image_paded / 255.

    # 6. 对真实边框的位置信息做相同转换操作
    if gt_boxes is None:
        return image_paded
    else:
        # 坐标相同转换方式
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image


def bboxes_iou(boxes1, boxes2):
    """
    计算边框的IoU的值
    :param boxes1: 格式为[..., 4], 4这个值为: [x_left, y_top, x_right, y_bottom]
    :param boxes2: 格式为[..., 4], 4这个值为: [x_left, y_top, x_right, y_bottom]
    :return:
    """
    # 1. 数据转换
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    # 2. 计算各个区域的面积
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 3. 获取左上角的最大值坐标以及右小角的最小值坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 4. 当且仅当右下角的坐标值大于左上角的坐标值的时候，存在交叉区域，计算交叉区域面积
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    # 5. 计算总面积
    union_area = boxes1_area + boxes2_area - inter_area

    # 6. 计算IoU的值(防止为零)，进行设置
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def read_pb_return_tensors(graph, pb_file, return_elements):
    """
    pb格式的模型恢复(加载对应的Tensor值)
    :param graph:
    :param pb_file:
    :param return_elements:
    :return:
    """
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    # 1. 获取类别列表
    classes_in_img = list(set(bboxes[:, 5]))

    # 2. 定义最优边框列表
    best_bboxes = []

    # 3.遍历所有类别
    for cls in classes_in_img:
        # a. 获取对应类别的边框
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        # b. 对当前类别的所有边框遍历处理
        while len(cls_bboxes) > 0:
            # 1. 获取概率值最大的边框，并添加到列表中
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)

            # 2. 获取剩下的边框
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])

            # 3. 计算最有边框和剩下边框的IoU的值
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])

            # 4. 计算权重系数向量
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                # IoU低于阈值的，权重系数设置为0
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                # 进行rbf权重系数更改操作
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            # 5. 更新边框置信度的值
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight

            # 6.获取置信度大于0的边框，循环进行处理
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    print(np.max(scores))
    score_mask = scores >= score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
