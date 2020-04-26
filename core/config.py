#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
# ================================================================

from easydict import EasyDict as edict
import os

__C = edict()
# Consumers can get config by: from config import cfg
data_name = 'voc'  # 给定是哪一个数据集! 可选 'plate' 'voc' or 'coco'


# ================================================================
# YOLO options
# ================================================================
__C.YOLO = edict()

# 给定数据标签所在的文件路径，要求每行一个类别
__C.YOLO.CLASSES = os.path.join('./data/classes/', data_name + '.names')

# 给定Anchor Box边框大小数据所在的文件路径，要求所有数据在同一行。第一个数字为宽度、第二个为高度，并且给定的是Feature Map上的大小，在原图中会乘以一个step
__C.YOLO.ANCHORS = "./data/anchors/basline_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY = 0.9995
# 给定提取Anchor Box的Feature Map的缩放因子/大小
__C.YOLO.STRIDES = [8, 16, 32]
# 各个Feature Map上每个锚点对应的Anchor Box的数目
__C.YOLO.ANCHOR_PER_SCALE = 3
# 计算损失函数的时候，IoU阈值，当IoU的大于阈值的时候认为属于正样本，小于该值的时候认为属于负样本
__C.YOLO.IOU_LOSS_THRESH = 0.5
# 难负样本选择过程中选择的阈值，在构建损失函数的时候，将背景区域并且预测IoU超过0.3的区域添加到损失函数中
__C.YOLO.HARD_NEGATIVE_IOU_LOSS_THRESH = 0.3
# YOLO v3中FPN模块中，上采样的方式: resize、deconv
__C.YOLO.UPSAMPLE_METHOD = "resize"
__C.YOLO.ORIGINAL_WEIGHT = "./checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT = "./checkpoint/yolov3_coco_demo.ckpt"


# ================================================================
# Train options
# ================================================================
__C.TRAIN = edict()

# 训练数据所在的文件路径
__C.TRAIN.ANNOT_PATH = os.path.join('./data/dataset/', data_name + '_train.txt')

# 每个批次的样本数目
__C.TRAIN.BATCH_SIZE = 2
#  每个批次训练模型用的图像大小
# __C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE = [416]

# 训练的时候是否做数据增强
__C.TRAIN.DATA_AUG = True
# 初始的更新的学习率
__C.TRAIN.LEARN_RATE_INIT = 1e-4
# 最终的学习率
__C.TRAIN.LEARN_RATE_END = 1e-8
# 阈值，表示在前2个epoch中学习率是增大的，之后学习率是减小的
__C.TRAIN.WARMUP_EPOCHS = 2
# 第一阶段的训练epoch的数目，仅更新最后一层的参数
__C.TRAIN.FISRT_STAGE_EPOCHS = 20
# 第二阶段的训练epoch的数目，训练所有参数
__C.TRAIN.SECOND_STAGE_EPOCHS = 200
# train训练过程中，如果需要做模型恢复，模型恢复的文件路径
__C.TRAIN.INITIAL_WEIGHT = "./checkpoint/yolov3_voc_test.ckpt"


# ================================================================
# TEST options
# ================================================================
__C.TEST = edict()

# 给定验证数据所在的文件路径
__C.TEST.ANNOT_PATH = os.path.join('./data/dataset/', data_name + '_test.txt')

# 做图像验证的时候，批次大小
__C.TEST.BATCH_SIZE = 2
# 做图像验证/预测的时候，输入的图像会resize成的大小是多少
__C.TEST.INPUT_SIZE = 500
# 图像预测/验证的时候，是否做数据增强
__C.TEST.DATA_AUG = False
# 图像验证过程中输出的相关参数
__C.TEST.WRITE_IMAGE = True
__C.TEST.WRITE_IMAGE_PATH = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
# 模型恢复的文件路径
# __C.TEST.WEIGHT_FILE = "./checkpoint/yolov3_coco.ckpt"
# 车牌检测模型恢复路径
__C.TEST.WEIGHT_FILE = "./checkpoint/yolov3_test_loss=0.6405.ckpt-200"

# VOC2007数据检测的模型恢复路径
# __C.TEST.WEIGHT_FILE = "./checkpoint/yolov3_test_loss=40.6538.ckpt-14"
# 图像可视化的时候是否可视化标签
__C.TEST.SHOW_LABEL = True
# 当置信度超过该值的时候，认为属于物体边框
__C.TEST.SCORE_THRESHOLD = 0.5
# 当两个预测边框的IoU超过该值的时候，进行NMS删除的操作
__C.TEST.IOU_THRESHOLD = 0.45

cfg = __C
