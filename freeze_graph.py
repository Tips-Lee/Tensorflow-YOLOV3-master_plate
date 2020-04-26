#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : freeze_graph.py
#   Author      : YunYang1994
#   Created date: 2019-03-20 15:57:33
#   Description :
#
# ================================================================

"""
如果需要将模型转换为pb模型的话，保证以下几点：
    -1. 必须模型恢复(网络结构、模型参数必须是一致的)
    -2. 必须给定网络的输入的Tensor和输出Tensor的字符串名称，是一个集合
NOTE:
    检查pb文件是否正常：通过netron这个应用来检查文件是否正常
    pip install netron # 安装
    netron # 命令行执行启动服务
    默认访问：http://localhost:8080
"""

import tensorflow as tf
from core.yolov3 import YOLOV3

# ckpt_file = "./checkpoint/yolov3_coco_demo.ckpt"
# pb_file = "./yolov3_coco.pb"
ckpt_file = "./checkpoint/yolov3_test_loss=0.6405.ckpt-200"
pb_file = "./yolov3_plate_test.pb"
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, name='input_data')

model = YOLOV3(input_data, trainable=False)
print(input_data, model.pred_sbbox, model.pred_mbbox, model.pred_lbbox)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                   input_graph_def=sess.graph.as_graph_def(),
                                                                   output_node_names=output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
