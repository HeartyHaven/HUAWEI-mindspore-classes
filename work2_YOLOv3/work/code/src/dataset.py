# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


"""YOLOv3 dataset"""
from __future__ import division

import os
from xml.dom.minidom import parse
import xml.dom.minidom

import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
import mindspore.dataset as de
from mindspore.mindrecord import FileWriter
import mindspore.dataset.vision.c_transforms as C
from src.config import ConfigYOLOV3ResNet18

# 读取xml文件标签，返回带有指定标签名的对象的集合
def xy_local(collection,element):
    xy = collection.getElementsByTagName(element)[0]
    xy = xy.childNodes[0].data
    return xy



def filter_valid_data(image_dir):
    """在image_dir和anno_path中过滤有效的图像"""
    
    label_id={'person':0, 'face':1, 'mask':2}
    all_files = os.listdir(image_dir)

    image_dict = {}
    image_files=[]
    for i in all_files:
        if (i[-3:]=='jpg' or i[-4:]=='jpeg') and i not in image_dict:
            image_files.append(i)
            label=[]
            xml_path = os.path.join(image_dir,i[:-3]+'xml')
            
            if not os.path.exists(xml_path):
                label=[[0,0,0,0,0]]
                image_dict[i]=label
                continue
            DOMTree = xml.dom.minidom.parse(xml_path)
            collection = DOMTree.documentElement
            # 在集合中获取所有框
            object_ = collection.getElementsByTagName("object")
            for m in object_:
                temp=[]
                name = m.getElementsByTagName('name')[0]
                class_num = label_id[name.childNodes[0].data]
                bndbox = m.getElementsByTagName('bndbox')[0]
                xmin = xy_local(bndbox,'xmin')
                ymin = xy_local(bndbox,'ymin')
                xmax = xy_local(bndbox,'xmax')
                ymax = xy_local(bndbox,'ymax')
                temp.append(int(xmin))
                temp.append(int(ymin))
                temp.append(int(xmax))
                temp.append(int(ymax))
                temp.append(class_num)
                label.append(temp)
            image_dict[i]=label
    return image_files, image_dict



def data_to_mindrecord_byte_image(image_dir, mindrecord_dir, prefix, file_num):
    """通过image_dir和anno_path创建MindRecord文件"""
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)
    image_files, image_anno_dict = filter_valid_data(image_dir)

    yolo_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
        "file": {"type": "string"},
    }
    writer.add_schema(yolo_json, "yolo_json")

    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        with open(image_path, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name],dtype=np.int32)
        #print(annos.shape)
        row = {"image": img, "annotation": annos, "file": image_name}
        writer.write_raw_data([row])
    writer.commit()



def preprocess_fn(image, box, file, is_training):
    """数据集预处理函数."""
    config_anchors = []
    temp = ConfigYOLOV3ResNet18.anchor_scales
    for i in temp:
        config_anchors+=list(i)
    
    anchors = np.array([float(x) for x in config_anchors]).reshape(-1, 2)
    do_hsv = False
    max_boxes = ConfigYOLOV3ResNet18._NUM_BOXES   # 最多框，50
    num_classes = ConfigYOLOV3ResNet18.num_classes   # 类别数，3

    # 随机数
    def _rand(a=0., b=1.):
        return np.random.rand() * (b - a) + a

    # true_boxes为函数_data_aug的结果box_data
    def _preprocess_true_boxes(true_boxes, anchors, in_shape=None):
        """获取True边界框"""
        # num_layers为锚点层数，本实验将锚点分为大框、中框、小框三个层次
        num_layers = anchors.shape[0] // 3
        # anchor_mask为锚点编号
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(in_shape, dtype='int32')

        # boxes_xy代表框的中心点坐标，boxes_wh代表框的宽高
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2.
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
       
        # 将输入boxes_xy和boxes_wh的值放缩到0-1范围内
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        # grid_shapes为网格，包含大网格（大框使用），中网格（中框使用），小网格（小框使用）
        # 大框尺寸为 32×32 ，中框尺寸为 16×16 ，小框尺寸为 8×8 
        # 网格维度分别为大框[11,20] ，中框[22,40] ，小框[44,80]
        grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]

        # y_true为已经映射到锚点、网格的框
        # y_true[0]为大框映射，维度为[11, 20, 3, 8] ，y_true[1]为中框映射，维度为[22, 40, 3, 8] ， y_true[2]为小框映射，维度为[44, 80, 3, 8] 
        # 其中3代表每个层次有几个框，本实验大框、中框、小框层都是有三个框对应
        # 其中8代表标签值，分别代表[boxes_xy,boxes_wh, Confidence(置信度)，class(one-hot值，person、face、mask)]
        y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]),
                            5 + num_classes), dtype='float32') for l in range(num_layers)]


        # 将锚点中心点放缩到（0，0）点，并扩展维度
        anchors = np.expand_dims(anchors, 0)
        # 得到anchors_max和anchors_min，维度为[1,9,2]
        # anchors_max 代表框中心点放缩到（0，0）点以后的右上角坐标
        anchors_max = anchors / 2.
        # anchors_min 代表框中心点放缩到（0，0）点以后的左下角坐标
        anchors_min = -anchors_max
        # 将boxes_wh中心点放缩到（0，0点）
        valid_mask = boxes_wh[..., 0] >= 1
        wh = boxes_wh[valid_mask]
        if len(wh) >= 1:
            # 扩展维度
            wh = np.expand_dims(wh, -2)
            # 得到boxes_max和boxes_min维度为[num_box,1,2]
            # boxes_max 代表框中心点放缩到（0，0）点以后的右上角坐标
            boxes_max = wh / 2.
            # boxes_min 代表框中心点放缩到（0，0）点以后的左下角坐标
            boxes_min = -boxes_max


            # 分别求每个框和9个锚点的iou值
            # iou为交并比，iou维度为[num_box, 9]，其中9代表9个锚点
            intersect_min = np.maximum(boxes_min, anchors_min)
            intersect_max = np.minimum(boxes_max, anchors_max)
            intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
            # intersect_area为交集面积
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]   
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            # 并集面积为(box_area + anchor_area - intersect_area)
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            # 对于每个框，选出最大iou值对应的锚点编号
            best_anchor = np.argmax(iou, axis=-1)
            # 将true_boxes映射到锚点编号、网格中
            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        # 将锚点和真实框放缩到（0，1），目的是保证其中心点统一
                        i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                        # 可以快速定位目标框属于哪个锚点
                        k = anchor_mask[l].index(n)

                        c = true_boxes[t, 4].astype('int32')
                        # y_true即数据预处理结果bbox_1、bbox_2、bbox_3，矩阵数据范围为0-1
                        y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                        y_true[l][j, i, k, 4] = 1.
                        y_true[l][j, i, k, 5 + c] = 1.

        pad_gt_box0 = np.zeros(shape=[ConfigYOLOV3ResNet18._NUM_BOXES, 4], dtype=np.float32)
        pad_gt_box1 = np.zeros(shape=[ConfigYOLOV3ResNet18._NUM_BOXES, 4], dtype=np.float32)
        pad_gt_box2 = np.zeros(shape=[ConfigYOLOV3ResNet18._NUM_BOXES, 4], dtype=np.float32)

        # gt_box 维度为[50,4]，存放了所有的y_true中的框，忽略网格和锚点信息
        # gt_box 即数据预处理结果gt_box1、gt_box2、gt_box3，矩阵数据范围为0-1。
        mask0 = np.reshape(y_true[0][..., 4:5], [-1])
        gt_box0 = np.reshape(y_true[0][..., 0:4], [-1, 4])
        gt_box0 = gt_box0[mask0 == 1]
        pad_gt_box0[:gt_box0.shape[0]] = gt_box0

        mask1 = np.reshape(y_true[1][..., 4:5], [-1])
        gt_box1 = np.reshape(y_true[1][..., 0:4], [-1, 4])
        gt_box1 = gt_box1[mask1 == 1]
        pad_gt_box1[:gt_box1.shape[0]] = gt_box1

        mask2 = np.reshape(y_true[2][..., 4:5], [-1])
        gt_box2 = np.reshape(y_true[2][..., 0:4], [-1, 4])
        gt_box2 = gt_box2[mask2 == 1]
        pad_gt_box2[:gt_box2.shape[0]] = gt_box2

        return y_true[0], y_true[1], y_true[2], pad_gt_box0, pad_gt_box1, pad_gt_box2

    # 图片和框resize，本实验将图片都统一为[352,640]进行训练和推理
    # 测试图片改变尺寸使用传统的先裁剪后resize方式，在保证图片不失真的情况下改变图片尺寸
    def _infer_data(img_data, input_shape, box):
        w, h = img_data.size # 真实边
        input_h, input_w = input_shape  # 设定边（[352,640]）
        # 变量scale为真实边和设定边的最小比例。为了保证图片不失真，长宽采取相同的放缩比例。
        scale = min(float(input_w) / float(w), float(input_h) / float(h))
        nw = int(w * scale)
        nh = int(h * scale)
        img_data = img_data.resize((nw, nh), Image.BICUBIC)

        # 将图片resize为(nw, nh)后，与预设定图片尺寸（[352,640]）不同，需要进行填充。
        new_image = np.zeros((input_h, input_w, 3), np.float32)

        # 填充采用两边填充，即将resize后的(nw, nh)大小图片放在（[352,640]）中间
        # 两边填充像素值128
        new_image.fill(128)
        img_data = np.array(img_data)
        if len(img_data.shape) == 2:
            img_data = np.expand_dims(img_data, axis=-1)
            img_data = np.concatenate([img_data, img_data, img_data], axis=-1)

        dh = int((input_h - nh) / 2)
        dw = int((input_w - nw) / 2)
        new_image[dh:(nh + dh), dw:(nw + dw), :] = img_data
        new_image /= 255.
        new_image = np.transpose(new_image, (2, 0, 1))
        new_image = np.expand_dims(new_image, 0)
        # 测试框不需要做任何处理
        return new_image, np.array([h, w], np.float32), box

    def _data_aug(image, box, is_training, jitter=0.3, hue=0.1, sat=1.5, val=1.5, image_size=(352, 640)):
        
        """数据增强函数."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # 变量iw, ih为原始图片大小
        iw, ih = image.size
        ori_image_shape = np.array([ih, iw], np.int32)
        # 变量h,w为设定图片大小（[352,640]）
        h, w = image_size

        if not is_training:
            return _infer_data(image, image_size, box)

        flip = _rand() < .5
        
        # box_data维度为[50,5]
        # 其中50代表每张图片最多框数设定，5代表[xmin,ymin,xmax,ymax,class]
        box_data = np.zeros((max_boxes, 5))
        flag =0
        
        while True:
            # 防止所有框都被清除
            # 变量jitter控制随机噪声大小
            # 带有噪声的图片(nw, nh)在一定范围内失真，失真比例由jitter控制
            new_ar = float(w) / float(h) * _rand(1 - jitter, 1 + jitter) / \
                     _rand(1 - jitter, 1 + jitter)
            # 变量scale为尺度的意思，在（0.25，2）比例尺度范围内获取多尺度特征
            scale = _rand(0.25, 2)

            # 通过改变new_ar从而改变resize后的图片大小(nw, nh)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)

            # 变量dx和dy代表噪声大小
            dx = int(_rand(0, w - nw))
            dy = int(_rand(0, h - nh))
            flag = flag + 1
            
            # 对图片和框进行相同的resize操作
            # 将大小为(nw, nh)的图片填充到（w,h）的dx,dy位置
            if len(box) >= 1:
                t_box = box.copy()
                np.random.shuffle(t_box)
                t_box[:, [0, 2]] = t_box[:, [0, 2]] * float(nw) / float(iw) + dx
                t_box[:, [1, 3]] = t_box[:, [1, 3]] * float(nh) / float(ih) + dy
                if flip:
                    t_box[:, [0, 2]] = w - t_box[:, [2, 0]]
                t_box[:, 0:2][t_box[:, 0:2] < 0] = 0
                t_box[:, 2][t_box[:, 2] > w] = w
                t_box[:, 3][t_box[:, 3] > h] = h
                box_w = t_box[:, 2] - t_box[:, 0]
                box_h = t_box[:, 3] - t_box[:, 1]
                t_box = t_box[np.logical_and(box_w > 1, box_h > 1)]  # 去掉无效框

            if len(t_box) >= 1:
                box = t_box
                break

        # 得到的变量box_data需要进一步传入函数_preprocess_true_boxes中进行锚点和网格映射
        box_data[:len(box)] = box
        # 调整图像大小
        image = image.resize((nw, nh), Image.BICUBIC)
        # 替换图像，其他位置用128填充
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        # 得到的变量images为预处理结果，可以直接输入网络训练
        image = new_image

        # 是否翻转图像
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 是否转灰度图
        gray = _rand() < .25
        if gray:
            image = image.convert('L').convert('RGB')

        # 当图像通道数为1
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        # 扭曲图像
        hue = _rand(-hue, hue)
        sat = _rand(1, sat) if _rand() < .5 else 1 / _rand(1, sat)
        val = _rand(1, val) if _rand() < .5 else 1 / _rand(1, val)
        image_data = image / 255.
        if do_hsv:
            x = rgb_to_hsv(image_data)
            x[..., 0] += hue
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x > 1] = 1
            x[x < 0] = 0
            image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
        image_data = image_data.astype(np.float32)

        # 预处理边界框
        bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 =             _preprocess_true_boxes(box_data, anchors, image_size)

        return image_data, bbox_true_1, bbox_true_2, bbox_true_3,                ori_image_shape, gt_box1, gt_box2, gt_box3

    if is_training:
        images, bbox_1, bbox_2, bbox_3, image_shape, gt_box1, gt_box2, gt_box3 = _data_aug(image, box, is_training)
        return images, bbox_1, bbox_2, bbox_3, gt_box1, gt_box2, gt_box3

    images, shape, anno = _data_aug(image, box, is_training)
    return images, shape, anno, file




def create_yolo_dataset(mindrecord_dir, batch_size=32, repeat_num=1, device_num=1, rank=0,
                        is_training=True, num_parallel_workers=8):
    """使用MindDataset创建YOLOv3数据集"""
    ds = de.MindDataset(mindrecord_dir, columns_list=["image", "annotation","file"], num_shards=device_num, shard_id=rank,
                        num_parallel_workers=num_parallel_workers, shuffle=is_training)
    decode = C.Decode()
    ds = ds.map(operations=decode, input_columns=["image"])
    compose_map_func = (lambda image, annotation, file: preprocess_fn(image, annotation,file, is_training))

    if is_training:
        hwc_to_chw = C.HWC2CHW()
        ds = ds.map(operations=compose_map_func, input_columns=["image", "annotation","file"],
                    output_columns=["image", "bbox_1", "bbox_2", "bbox_3", "gt_box1", "gt_box2", "gt_box3"],
                    # column_order=["image", "bbox_1", "bbox_2", "bbox_3", "gt_box1", "gt_box2", "gt_box3"],
                    num_parallel_workers=num_parallel_workers)
        ds = ds.map(operations=hwc_to_chw, input_columns=["image"], num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.repeat(repeat_num)
    else:
        ds = ds.map(operations=compose_map_func, input_columns=["image", "annotation","file"],
                    output_columns=["image", "image_shape", "annotation","file"],
                    # column_order=["image", "image_shape", "annotation","file"],
                    num_parallel_workers=num_parallel_workers)
    return ds






