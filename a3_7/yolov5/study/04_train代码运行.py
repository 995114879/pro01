# -*- coding: utf-8 -*-
import train


def t0():
    train.run()  # 直接运行


def t1():
    train.run(
        cfg="../models/yolov5s_copy_01.yaml",
        exclude=10,  # 模型迁移的时候，[10,) 不进行模型参数迁移
        freeze=[10]  # 模型迁移/模型恢复的时候, [0,10) 进行参数冻结，不参与模型训练
    )


if __name__ == '__main__':
    t1()
