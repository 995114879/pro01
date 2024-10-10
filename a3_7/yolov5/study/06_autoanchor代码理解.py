# -*- coding: utf-8 -*-
import yaml

from models.yolo import Model, Detect
from utils.autoanchor import check_anchors, kmean_anchors
from utils.dataloaders import create_dataloader
from utils.general import colorstr


def t0(dataset, model, hyp):
    imgsz = 640
    net = model.model[-1]
    print(f"更新前:{net.anchors}")
    # 内部核心：
    # 1. 如何评估anchor box的尺度是好的还是不好的?  ---> 比较/统计真实边框和先验框之间的高&宽比值, 统计比值在[1/thr,thr]之间的比例，以这个比例衡量先验框尺度的好坏
    # 2. 如何选择一个更好的anchor box尺度? ---> 针对给定数据中的所有边框的高度、宽度进行聚类处理，从而选择出蔟中心
    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
    print(f"更新后:{net.anchors}")


def t1(dataset):
    for na in [1, 3, 6, 9, 12, 15, 18]:
        print("=" * 100)
        anchors = kmean_anchors(dataset, n=na, img_size=640, thr=4, gen=1000, verbose=False)
        print(anchors.reshape(-1))


def run():
    hyp = "../data/hyps/hyp.scratch-low_copy_00.yaml"
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    cfg_path = "../models/yolov5s_copy_04.yaml"  # 给定模型配置文件路径
    model = Model(
        cfg=cfg_path,  # 给定模型配置文件或者dict字典
        ch=3,  # 输入的通道数目
        nc=None,  # 类别数目， 不给定的时候，直接cfg里面的nc
        anchors=None  # 给定初始的先验框尺度(高度/宽度)
    )

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

    # t0(dataset, model, hyp)
    t1(dataset)


if __name__ == '__main__':
    run()
