# -*- coding: utf-8 -*-
import random

import yaml

import numpy as np
import cv2 as cv

from utils.augmentations import mixup, random_perspective
from utils.dataloaders import create_dataloader
from utils.general import colorstr


def t0(hyp):
    train_path = r"/mnt/code/shenlan/detection/datasets/coco128/images/train2017"
    train_loader, dataset = create_dataloader(
        train_path,  # 训练数据所在的文件夹路径 yaml中配置给定的
        640,  # 图像尺度大小
        4,  # 批次大小
        32,  # 模型前向过程中，最小一个feature map和原始图像之间的缩放比 --> 最大缩放比
        False,  # 是否是单一类别的目标检测
        hyp=hyp,  # 超参数
        augment=True,  # 做不做数据增强
        cache=None,  # 数据缓存方式
        rect=False,
        rank=-1,  # 多GPU时候使用
        workers=0,  # 数据加载的线程数目
        image_weights=False,  # 是否做目标检测的类别权重的计算 --> 是否基于不同类别的权重，针对不同类别进行数据抽样
        quad=False,
        prefix=colorstr('train: '),
        shuffle=True  # 数据是否打乱顺序
    )

    for batch_image, batch_labels, batch_im_files, batch_shapes in train_loader:
        print("=" * 50)
        print(batch_image.shape)  # x 图像的对象 [batch_size, 3, H, W]
        #  M表示当前批次中的总边框数目 6分别为image_idx、label_class_idx、cx、cy、w、h  后面四个是百分比的值
        print(batch_labels.shape)  # y 图像的标签 [M,6]
        print(batch_labels)
        print(batch_im_files)  # 当前批次对应的图像路径
        print(batch_shapes)  # 图像大小
        break


def t1(hyp):
    train_path = r"/mnt/code/shenlan/detection/datasets/coco128/images/train2017"
    train_loader, dataset = create_dataloader(
        train_path,  # 训练数据所在的文件夹路径 yaml中配置给定的
        320,  # 图像尺度大小
        4,  # 批次大小
        32,  # 模型前向过程中，最小一个feature map和原始图像之间的缩放比 --> 最大缩放比
        False,  # 是否是单一类别的目标检测
        hyp=hyp,  # 超参数
        augment=True,  # 做不做数据增强
        cache=None,  # 数据缓存方式
        rect=False,
        rank=-1,  # 多GPU时候使用
        workers=0,  # 数据加载的线程数目
        image_weights=False,  # 是否做目标检测的类别权重的计算 --> 是否基于不同类别的权重，针对不同类别进行数据抽样
        quad=False,
        prefix=colorstr('train: '),
        shuffle=True  # 数据是否打乱顺序
    )

    img1, labels1 = dataset.load_mosaic(3, augment=False)
    print(labels1)
    for label, x1, y1, x2, y2 in labels1:
        cv.rectangle(img1, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(0, 0, 255), thickness=2)
    img, labels = mixup(img1, labels1, *dataset.load_mosaic(5, augment=False))
    print(labels)

    cv.imshow('img1', img1)
    cv.imshow('img', img)
    cv.waitKey(-1)
    cv.destroyAllWindows()


def t2(hyp):
    train_path = r"/mnt/code/shenlan/detection/datasets/coco128/images/train2017"
    train_loader, dataset = create_dataloader(
        train_path,  # 训练数据所在的文件夹路径 yaml中配置给定的
        320,  # 图像尺度大小
        4,  # 批次大小
        32,  # 模型前向过程中，最小一个feature map和原始图像之间的缩放比 --> 最大缩放比
        False,  # 是否是单一类别的目标检测
        hyp=hyp,  # 超参数
        augment=True,  # 做不做数据增强
        cache=None,  # 数据缓存方式
        rect=False,
        rank=-1,  # 多GPU时候使用
        workers=0,  # 数据加载的线程数目
        image_weights=False,  # 是否做目标检测的类别权重的计算 --> 是否基于不同类别的权重，针对不同类别进行数据抽样
        quad=False,
        prefix=colorstr('train: '),
        shuffle=True  # 数据是否打乱顺序
    )

    img1, labels1 = dataset.load_mosaic(3, augment=False)

    C = np.asarray([
        [1, 0, -320],
        [0, 1, -320],
        [0, 0, 1]
    ], dtype=np.float32)
    img2 = cv.warpAffine(img1, C[:2], dsize=(640, 640), borderValue=(114, 114, 114))

    # Rotation and Scale 旋转+缩放
    R = np.eye(3)
    a = random.uniform(-30, 30)  # 随机一个选择系数
    s = random.uniform(1 - 0.5, 1 + 0.5)  # 随机缩放的系数
    R[:2] = cv.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)  # 获取旋转、缩放的矩阵
    img3 = cv.warpAffine(img2, R[:2], dsize=(320, 320), borderValue=(114, 114, 114))

    M1 = R @ C  # 相当于先执行C，然后再执行R
    M2 = C @ R
    # warpAffine计算规则：src(x, y) = dst(m11 * x + m12 * y + m13, m21 * x + m22 * y + m23)
    # 原始图像中的坐标(5,8) --映射--> 新图像的坐标: (m11 * 5 + m12 * 8 + m13, m21 * 5 + m22 * 8 + m23)
    # [[5,8,1]] * [[m11,m12,m13],[m21,m22,m23]].T
    # 或者：[[m11,m12,m13],[m21,m22,m23]] * [[5,8,1]].T
    img41 = cv.warpAffine(img1, M1[:2], dsize=(320, 320), borderValue=(114, 114, 114))
    img42 = cv.warpAffine(img1, M2[:2], dsize=(320, 320), borderValue=(114, 114, 114))

    img, labels4 = random_perspective(
        img1,
        labels1,
        [],
        degrees=hyp['degrees'],
        translate=hyp['translate'],
        scale=hyp['scale'],
        shear=hyp['shear'],
        perspective=hyp['perspective'],
        border=dataset.mosaic_border
    )  # border to remove
    print(labels4)

    cv.imshow('img1', img1)
    cv.imshow('img2', img2)
    cv.imshow('img3', img3)
    cv.imshow('img41', img41)
    cv.imshow('img42', img42)
    cv.imshow('img', img)
    cv.waitKey(-1)
    cv.destroyAllWindows()


def run():
    hyp = "../data/hyps/hyp.scratch-low_copy_00.yaml"
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    # t0(hyp)
    t1(hyp)
    # t2(hyp)


if __name__ == '__main__':
    run()
