# -*- coding: utf-8 -*-
import torch
import yaml
import torch.nn as nn
from torch.optim import lr_scheduler

from models.yolo import Model
from utils.general import one_cycle
from utils.torch_utils import smart_optimizer

import matplotlib.pyplot as plt


def t0(model, hyp):
    optimizer = 'SGD'
    """
    优化器：深度学习中进行参数更新的代码/方式，其实所有的优化器都是基于梯度的优化，本质上都是BP的过程
    不同优化器，只是在求解更新值的有所区别:
        常规的优化器/SGD: theta = theta - lr*gd, theta就是待更新的参数，lr是更新的学习率，gd就是损失loss关于参数theta的对应梯度值
        包含L2惩罚项的SGD优化公式: theta = theta - lr*gd - weight_decay*theta
    
    模型过拟合: 模型在训练数据上效果很好/预测正确，但是在测试/线上预测效果不好/预测不对；
        ---> 单纯考虑线性回归模型: pred_y = w * x + b
        ---> 效果不好也就是pred_y和真实值差值非常大
        ---> 训练数据集有一个样本特征: [0.1, 0.2, -0.3]，真实值和预测值都是5
        ---> 当前的测试样本特征: [0.1,0.4,-0.3], 真实值是4，但是预测值是205
    """
    optimizer = smart_optimizer(model, optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    print(optimizer)


def run():
    hyp = "../data/hyps/hyp.scratch-low_copy_00.yaml"
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    cfg_path = "../models/yolov5s_copy_00.yaml"  # 给定模型配置文件路径
    model = Model(
        cfg=cfg_path,  # 给定模型配置文件或者dict字典
        ch=3,  # 输入的通道数目
        nc=None,  # 类别数目， 不给定的时候，直接cfg里面的nc
        anchors=None  # 给定初始的先验框尺度(高度/宽度)
    )

    t0(model, hyp)


def t1():
    epochs = 100

    net = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )
    g = [[], []]
    for name, param in net.named_parameters():
        print(name, param.shape)
        if 'bias' in name:
            g[0].append(param)
        else:
            g[1].append(param)

    optimizer = torch.optim.SGD(params=[{'params': g[0], 'name': 'g0'}], lr=0.005, momentum=0.0, nesterov=False,
                                weight_decay=0.0)
    optimizer.add_param_group({"params": g[1], "lr": 0.01, "weight_decay": 0.002, 'name': 'g1'})
    lrf = 0.01
    lf = lambda cs: (1 - cs / epochs) * (1.0 - lrf) + lrf  # linear cs 其实就是每调用一次，这个值就增加1
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    loss_fn = nn.MSELoss()

    _x = torch.rand(5, 2)
    _y = torch.rand(5, 1)

    g_lrs = {'g0': [], 'g1': []}
    for i in range(epochs + 1):
        print("=" * 100)
        print(f"epochs:{i + 1}")

        # 前向过程
        _o = net(_x)
        _loss = loss_fn(_o, _y)
        print(_loss)

        # 反向过程
        optimizer.zero_grad()
        _loss.backward()
        # print(net[-1].weight.view(-1), net[-1].weight.grad.view(-1))
        # print(net[-1].bias, net[-1].bias.grad)
        for group in optimizer.param_groups:
            print(group['name'], "  ", f"lr:{group['lr']}")
            g_lrs[group['name']].append(group['lr'])
        optimizer.step()  # 更新
        # print(net[-1].weight.view(-1), net[-1].weight.grad.view(-1))
        # print(net[-1].bias, net[-1].bias.grad)

        if i % 3 == 0:
            scheduler.step()  # 学习率更新

    plt.plot(range(len(g_lrs['g1'])), g_lrs['g0'])
    plt.plot(range(len(g_lrs['g1'])), g_lrs['g1'])
    plt.show()


def t2():
    base_lr = 0.5
    epochs = 100
    lrf = 0.2
    lf = lambda cs: (1 - cs // 10 * 10 / epochs) * (1.0 - lrf) + lrf  # linear cs 其实就是每调用一次，这个值就增加1
    lf = one_cycle(1, lrf, epochs)  # cosine 1->hyp['lrf']

    x = list(range(epochs))
    lrs = [lf(current_step) * base_lr for current_step in x]

    plt.plot(x, lrs)
    plt.show()


if __name__ == '__main__':
    t2()
