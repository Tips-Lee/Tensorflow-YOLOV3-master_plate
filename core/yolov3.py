#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolov3.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 10:47:03
#   Description :
#
# ================================================================

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""

    def __init__(self, input_data, trainable):

        self.trainable = trainable  # 输入的占位符，用于表示模型是训练还是预测，训练为True，预测为False
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)  # 下标索引和类别名称之间的映射关系，索引从0开始
        self.num_class = len(self.classes)  # 总类别数目
        self.strides = np.array(cfg.YOLO.STRIDES)  # 各个Feature Map上的缩放比例
        self.anchors = utils.get_anchors(cfg.YOLO.ANCHORS)  # 各个Feature Map上的Anchor Box的W,H比例值
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE  # 各个Feature Map上的Anchor的数目
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH  # IoU的阈值，默认为0.5
        self.hard_negative_iou_loss_thresh = cfg.YOLO.HARD_NEGATIVE_IOU_LOSS_THRESH
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD  # YOLOv3中数据上采样的方式，可选值:resize、deconv

        # 1. 构建YOLO v3模型对象
        try:
            # 大物体检测的前向预测结果、中等物体的前向检测结果、小物体的前向检测结果
            # 形状均为: [N, H, W, anchor_per_scale * （num_class + 5）]
            # 表示的含义是：这里有N个图像，每个图像划分为H*W个GridCell,
            # 每个GridCell中预测anchor_per_scale个边框，每个边框预测属于各个类别的置信度、边框坐标信息以及是否包含物体的概率值。
            # 默认Voc数据集中，anchor_per_scale为3, num_class为20
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_nework(input_data)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        with tf.variable_scope('pred_sbbox'):  # 第3层的预测结果构建(小物体的检测)
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):  # 第4层的预测结果构建(中等物体的检测)
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):  # 第5层的预测结果构建(大物体的检测)
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def __build_nework(self, input_data):
        """
        基于输入的数据构建YOLO v3模型
        :param input_data:
        :return:
        """
        # 1. 构建darknet53网络结构，得到输出值(第一个分支、第二个分支以及最后输出数据)
        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)  # 骨干网络(Encoder网络)

        # 2. 对最终输出数据连续5次卷积操作([3,3,1024] -> [1,1,512] -> [3,3,1024] -> [1,1,512]-> [3,3,1024] -> [1,1,512])
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv52')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv54')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')

        # 3. 最后一个分支对应的预测值输出
        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, self.anchor_per_scale * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        # 4. 将最后一个分支进行卷积转换+上采样
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        # 5. 将最后一个分支的数据和第二个分支的数据合并
        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        # 6. 对第二个分支数据连续5次卷积操作
        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        # 7. 第二个分支对应的预测值输出
        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, name='conv_mobj_branch')
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        # 8. 将第二个分支的数据进行卷积转换 + 上采样
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        # 9. 将第二个分支的数据和第三个分支的数据合并
        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        # 10. 对第三个分支的数据连续5次卷积操作
        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        # 11. 第三个分支对应的预测值输出
        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        # 12. 返回三个分支的数据输出，也就是大目标、中等目标、小目标的检测
        return conv_lbbox, conv_mbbox, conv_sbbox

    def decode(self, conv_output, anchors, stride):
        """
        :param conv_output: 网络对应层的前向预测结果，[batch_size, feature_map_h, feature_map_w, 75]; 75=3*(5+20)
        :param anchors: 对应层的Anchor Box的大小信息, eg: 第三层默认为: [[1.25  1.625] [2.    3.75 ] [4.125 2.875]]
        :param stride: 对应层的Anchor Box的步长信息，eg：第三层为:8
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
               output_size = feature_map_h = feature_map_w
        """
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)  # 每个锚点Anchor Box的数目，默认3个

        # 前向预测结果做一个shape变化， [N, H, W, anchor_per_scale * (5+num_class)] -> [N, H, W, anchor_per_scale , 5+num_class]
        conv_output = tf.reshape(conv_output,
                                 (batch_size, output_size, output_size, anchor_per_scale,
                                  5 + self.num_class))  # [N, H, W, anchor_of_scale, anchor_infos]

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]  # 对于每个预测框而言，获取x和y的坐标预测值
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]  # 对于每个预测框而言，获取w和h的宽度、高度预测值
        conv_raw_conf = conv_output[:, :, :, :, 4:5]  # 对于每个预测框而言，获取预测为物体的概率的值
        conv_raw_prob = conv_output[:, :, :, :, 5:]  # 对于每个预测框而言，获取预测属于所有类别的概率值

        # tf.tile: 数据进行重复，按照给定的参数重复多少多少次，下面两行代码的主要功能是：生成Feature Map上的网格点
        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])  # Feature Map上的坐标点位置信息
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])  # Feature Map上的坐标点位置信息

        # 首先得到Feature Map上的网格点坐标[H,W,2] -> 做一个坐标的复制: [N, H, W, anchor_per_scale, 2] -> type类型的转换
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)  # Feature Map上的坐标点位置信息

        # 边框最终预测的中心点、宽度、高度坐标的计算
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride  # 类似YOLO v2中介绍的转换函数
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        # 使用sigmoid计算概率值
        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        # 结果返回: [N, H, W, 3, 25] NOTE: H == W
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        # boxes1: [N, H, W, 3, 4] -> N个图像，每个图像有H*W个Grid Cell, 每个Grid Cell预测3个边框，每个边框预测的坐标信息: [left_x, top_y, right_x, bottom_y]
        # boxes2: [N, H, W, 3, 4] -> 意义和预测边框boxes1意义，只是属于实际边框信息
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        # 防止前面代码计算出来的坐标大小不对，保证左上角的坐标点的值一定小于右下角的
        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        # 计算各个边框的面积， 形状: [N, H, W, 3]
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        # 计算IoU的重叠区域的坐标信息（左上角坐标选最大、右下角坐标选最小）
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        # 计算重叠区域的面积(面积一定大于0)， [N, H, W, 3]
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # 计算总的面积以及IoU的值
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        # 计算GIoU（左上角坐标选最小、右下角坐标选最大）
        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        # 计算大区域的面积(两个区域合并后的面积)
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        # 计算giou
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]  # N
        output_size = conv_shape[1]  # H 52

        input_size = stride * output_size  # 得到输入图像的大小
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]  # 是否是物体的置信度，没有经过概率转换的
        conv_raw_prob = conv[:, :, :, :, 5:]  # 具体是那个类别的物体置信度，没有经过概率转换的

        pred_xywh = pred[:, :, :, :, 0:4]  # 已经经过边框回归转换之后的坐标信息(最终预测边框坐标)
        pred_conf = pred[:, :, :, :, 4:5]  # 已经经过sigmoid转换之后的是否是物体的概率值

        label_xywh = label[:, :, :, :, 0:4]  # 得到实际目标属性对应的坐标信息(也即使AnchorBox对应的构建的实际值是什么)
        respond_bbox = label[:, :, :, :, 4:5]  # 得到属于物体的置信度, 这个边框如果预测物体为1，不预测物体为0
        label_prob = label[:, :, :, :, 5:]  # 得到属于各个类别的置信度

        # 计算预测边框和AnchorBox目标属性之间的GIoU的值，形状为: [N, H, W, 3, 1]
        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)  # 基于预测边框和实际边框的坐标计算GIOU的值

        # 得到真实边框对应的面积占比(2.0-真实边框面积/整个图像的面积)<为了最终体现小边框的损失更加重要>
        input_size = tf.cast(input_size, tf.float32)
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)  # 1 < bbox_loss_scale < 2
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        # 计算的预测边框和真实边框之间的IoU值, 形状为: [N, H, W, 3, M] N个图像，每个图像H*W个GridCell, 每个Cell中3个预测边框，每个预测边框和M个真实边框之间的IoU值
        # -> [N, H, W, 3] N个图像，每个图像H*W个GridCell, 每个Cell中3个预测边框，每个预测边框和所有真实边框的IoU的最大值
        # -> [N, H, W, 3, 1]
        # 将预测边框坐标信息做一个shape转换: [N, H, W, 3, 4] -> [N, H, W, 3, 1, 4]
        # 将真实边框坐标信息也做一个shape转换: [N, M, 4] -> [N, 1, 1, 1, M, 4]
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        # 计算包含物体和不包含物体的置信度损失函数
        # 计算背景的损失函数(那些区域属于背景)，属于背景为1，不属于为0 --> 得到那些预测失败的(IoU在0.5以上)背景 --> 难负样本
        # TODO: 自己思考一下，下面代码中原来的小于符号改为大于符号是否是正确的?
        # respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou > self.iou_loss_thresh, tf.float32)
        # 更改一个参数，专门用于难负样本挖掘的操作，hard_negative_iou_loss_thresh默认值为0.3
        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou > self.hard_negative_iou_loss_thresh, tf.float32)

        # 计算预测物体的损失函数(Focal部分)
        conf_focal = self.focal(respond_bbox, pred_conf)

        # 损失函数合并(关注物体区域以及预测失败的背景区域)
        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        # 计算预测物体的各个类别的损失函数
        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        # 求和 -> 求均值的损失函数值: [N, H, W, 3, 1] -> [N,] -> []
        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):  # 小边框的损失，也就是第三层的预测结果的损失
            loss_sbbox = self.loss_layer(
                self.conv_sbbox,  # 网络第三层的预测结果的输出，形状为: [N, H, W, 75], 75 = 3 * (4 + 1 + 20)
                self.pred_sbbox,  # 网络第三层的预测结果输出经过转换后的数据(转换是做坐标和概率转换), 形状为: [N, H, W, 3, 25] 25 = 4 + 1 + 20
                label_sbbox,  # 目标属性信息，形状为: [N,H,W,3,25]
                true_sbbox,  # 是图像上对应的边框坐标信息，形状为: [N, M, 4]，M为一个图像上的总边框数目
                anchors=self.anchors[0],  # 该Feature Map对应的Anchor Box的大小
                stride=self.strides[0]  # 该Feature Map上每个像素点对应的缩放比例，第三层为8
            )

        with tf.name_scope('medium_box_loss'):  # 中等边框的损失，也就是第四层的预测结果的损失
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):  # 大边框的损失，也就是第五层的预测结果的损失
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss
