# -*- coding: utf-8 -*-
import time

import numpy as np
import torch
from thop import profile
from torchvision import models


@torch.no_grad()
def t0():
    model_classes = [
        models.vgg16,
        models.alexnet,
        models.googlenet,
        models.resnet101,
        models.shufflenet_v2_x1_0,
        models.mobilenet_v2,
        models.mobilenet_v3_large,
        models.mobilenet_v3_small,
        models.densenet121
    ]
    input_ = torch.randn(1, 3, 224, 224)

    with open("model_params.txt", 'w', encoding="utf-8") as writer:
        def _print(_msg):
            print(_msg)
            writer.writelines(f"{_msg}\n")

        def _format(_v):
            _suffix_list = ['K', 'M', 'G']
            _suffix_idx = -1
            while _suffix_idx < len(_suffix_list) and _v > 1024:
                _v = _v / 1024.0
                _suffix_idx += 1
            return f"{_v:.3f}{_suffix_list[_suffix_idx]}" if _suffix_idx > 0 else f"{_v:.3f}"

        for model_cls in model_classes:
            _print("=" * 100)
            _print(model_cls.__name__)
            net = model_cls()
            net.eval()
            macs, params = profile(net, inputs=(input_,))
            print(params)
            _print(f"浮点运算量:{_format(macs)}Flops")
            _print(f"参数量:{_format(params)}")

            # batch_size_list = [1, 4, 8, 16]
            batch_size_list = [1, 4]
            for batch_size in batch_size_list:
                input_list = [
                    torch.rand(batch_size, 3, 224, 224) for _ in range(100)
                ]
                run_times = []
                for input_ in input_list:
                    st = time.time()
                    net(input_)
                    et = time.time()
                    run_times.append(et - st)

                _print(f"{batch_size}平均耗时:{np.mean(run_times[10:81])}")


@torch.no_grad()
def t1():
    net = models.shufflenet_v2_x1_0()
    net.eval()

    for batch_size in [1, 4, 8, 16]:
        input_list = [
            torch.rand(batch_size, 3, 224, 224) for _ in range(100)
        ]
        run_times = []
        for input_ in input_list:
            st = time.time()
            net(input_)
            et = time.time()
            run_times.append(et - st)

        print(f"{batch_size}平均耗时:{np.mean(run_times[10:81])}")


@torch.no_grad()
def t2():
    net = models.shufflenet_v2_x1_0()
    net.eval()

    n = 2000
    input_x = torch.randn(n, 3, 224, 224)
    for batch_size in [1, 4, 8, 16]:
        total_batch_size = n // batch_size
        total_times = 0
        for i in range(total_batch_size):
            input_x_ = input_x[i * batch_size:i * batch_size + batch_size, ...]
            st = time.time()
            net(input_x_)
            et = time.time()
            total_times += et - st
        print(f"{batch_size}总耗时:{total_times:.3f}")


if __name__ == '__main__':
    t2()
