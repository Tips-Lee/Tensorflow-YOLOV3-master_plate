# -- encoding:utf-8 --

import os
from xml.etree import ElementTree as ET

with open(r'.\data\dataset\plate_train.txt', 'w') as writer:
    root_dir = r".\data\plate"
    dir_path = os.path.join(root_dir, "JPEGImages")
    xml_dir_path = os.path.join(root_dir, "Annotations")
    files = os.listdir(xml_dir_path)
    print('create plain_train.txt begin!')
    for name in files:
        # 1. 构建路径
        xml_file = os.path.join(xml_dir_path, name)

        # 2. 构建数据读取
        tree = ET.parse(xml_file)

        # 3. 得到xml的根节点
        root = tree.getroot()

        # 4. 获取路径信息
        filename = root.find("filename").text

        # 5. 构建路径
        image_file_path = os.path.join(dir_path, filename)
        writer.writelines(image_file_path)

        # 6. 加载目标
        for obj in root.findall('object'):
            # a. 得到标签
            label = obj.find('name').text

            # b. 获取坐标信息
            bbox = obj.find("bndbox")
            xmin = bbox.find("xmin").text
            ymin = bbox.find("ymin").text
            xmax = bbox.find("xmax").text
            ymax = bbox.find("ymax").text
            # 这里输出为零的原因是：因为只有一个类别
            writer.writelines(" {},{},{},{},0".format(xmin, ymin, xmax, ymax))
        writer.writelines('\n')
    print('Done !!!')
