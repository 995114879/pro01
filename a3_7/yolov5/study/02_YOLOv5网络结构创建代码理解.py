# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
print(sys.path)

import torch

from models.yolo import Model


def t00():
    cfg_path = "../models/yolov5s_copy_00.yaml"  # 给定模型配置文件路径
    import yaml  # for torch hub
    yaml_file = Path(cfg_path).name
    with open(cfg_path, encoding='ascii', errors='ignore') as f:
        yaml = yaml.safe_load(f)  # model dict
    print(yaml)


def t0():
    cfg_path = "../models/yolov5s_copy_00.yaml"  # 给定模型配置文件路径
    net = Model(
        cfg=cfg_path,  # 给定模型配置文件或者dict字典
        ch=3,  # 输入的通道数目
        nc=None,  # 类别数目， 不给定的时候，直接cfg里面的nc
        anchors=None  # 给定初始的先验框尺度(高度/宽度)
    )
    print(net)

    x = torch.rand(2, 3, 608, 608)
    r = net(x)
    print(r)
    print(type(r))
    print(r[0].shape)
    print(r[1].shape)
    print(r[2].shape)

    torch.onnx.export(
        model=net,
        args=(x,),
        f='yolov5s_copy_00.onnx',
        input_names=['image'],
        output_names=['labels'],
        opset_version=12
    )


def t1():
    # 增加anchor box
    cfg_path = "../models/yolov5s_copy_01.yaml"  # 给定模型配置文件路径
    net = Model(
        cfg=cfg_path,  # 给定模型配置文件或者dict字典
        ch=3,  # 输入的通道数目
        nc=None,  # 类别数目， 不给定的时候，直接cfg里面的nc
        anchors=None  # 给定初始的先验框尺度(高度/宽度)
    )
    print(net)

    x = torch.rand(2, 3, 608, 608)
    r = net(x)
    print(r)
    print(type(r))
    print(len(r))
    for rr in r:
        print(rr.shape)

    torch.onnx.export(
        model=net,
        args=(x,),
        f='yolov5s_copy_00.onnx',
        input_names=['image'],
        output_names=['labels'],
        opset_version=12
    )


def t2():
    # 减少anchor box
    cfg_path = "../models/yolov5s_copy_02.yaml"  # 给定模型配置文件路径
    net = Model(
        cfg=cfg_path,  # 给定模型配置文件或者dict字典
        ch=3,  # 输入的通道数目
        nc=None,  # 类别数目， 不给定的时候，直接cfg里面的nc
        anchors=None  # 给定初始的先验框尺度(高度/宽度)
    )
    print(net)

    x = torch.rand(2, 3, 608, 608)
    r = net(x)
    print(r)
    print(type(r))
    print(len(r))
    for rr in r:
        print(rr.shape)

    torch.onnx.export(
        model=net,
        args=(x,),
        f='yolov5s_copy_00.onnx',
        input_names=['image'],
        output_names=['labels'],
        opset_version=12
    )


def t3():
    # 新增/更改一些模块：上采样模块、下采样模块
    cfg_path = "../models/yolov5s_copy_03.yaml"  # 给定模型配置文件路径
    net = Model(
        cfg=cfg_path,  # 给定模型配置文件或者dict字典
        ch=3,  # 输入的通道数目
        nc=None,  # 类别数目， 不给定的时候，直接cfg里面的nc
        anchors=None  # 给定初始的先验框尺度(高度/宽度)
    )
    print(net)

    x = torch.rand(2, 3, 608, 608)
    r = net(x)
    print(r)
    print(type(r))
    print(len(r))
    for rr in r:
        print(rr.shape)

    torch.onnx.export(
        model=net,
        args=(x,),
        f='yolov5s_copy_00.onnx',
        input_names=['image'],
        output_names=['labels'],
        opset_version=12
    )


def t4():
    cfg_path = "../models/yolov5s_copy_03.yaml"  # 给定模型配置文件路径
    net = Model(
        cfg=cfg_path,  # 给定模型配置文件或者dict字典
        ch=3,  # 输入的通道数目
        nc=None,  # 类别数目， 不给定的时候，直接cfg里面的nc
        anchors=None  # 给定初始的先验框尺度(高度/宽度)
    )
    print(net)

    x = torch.rand(2, 3, 608, 608)

    # 训练时候的返回值
    # na: number of anchor --> 每个锚点/grid对应几个anchor box/预测边框
    # nc: number of class --> 类别数目
    # N: 批次样本大小；H：feature map的高度；W：feature map的宽度；
    # 训练时候返回各个分支/各层对应的预测值, [N,na,H,W,nc+1+4]
    net.train()
    r = net(x)
    print(type(r))
    for rr in r:
        print(rr.shape)

    # 推理时候的返回值
    # 推理预测时候，返回的是预测结果，是一个二元组
    # 二元组的第一个元素，就是模型推理预测结果，tensor对象，shape为: [N, na*H*W, 4+1+nc], 每个样本、每个预测边框对应的中心点坐标、预测边框高度&宽度、置信度
    # 二元组的第二个元素，和训练时候返回的结果一样，list(tensor)的结构, tensor shape为: [N,na,H,W,nc+1+4]
    with torch.no_grad():
        print("=" * 100)
        net.eval()
        r = net(x)
        print(type(r))
        print(r[0].shape)


if __name__ == '__main__':
    t4()