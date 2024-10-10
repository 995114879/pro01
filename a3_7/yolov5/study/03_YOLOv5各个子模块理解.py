# -*- coding: utf-8 -*-

from models.common import *


def t0():
    x = torch.rand(2, 3, 100, 100)
    focus = Focus(c1=3, c2=64, k=3, s=1)
    r1 = focus(x)
    print(r1.shape)

    conv = Conv(c1=3, c2=64, k=6, s=2, p=2)
    r2 = conv(x)
    print(r2.shape)


def t1():
    x = torch.rand(2, 64, 100, 100)
    csp = BottleneckCSP(c1=64, c2=64, n=2)
    r1 = csp(x)
    print(r1.shape)

    c3 = C3(c1=64, c2=64, n=2)
    r2 = c3(x)
    print(r2.shape)


def t2():
    # 池化的计算量: kh * kw * H * W * C
    # spp中池化的计算量: 5*5*H*W*C + 9*9*H*W*C + 13*13*H*W*C =275*H*W*C
    # sppf中池化的计算量: 5*5*H*W*C + 5*5*H*W*C + 5*5*H*W*C = 75*H*W*C
    x = torch.rand(2, 64, 100, 100)

    spp = SPP(c1=64, c2=64)
    r1 = spp(x)
    print(r1.shape)

    sppf = SPPF(c1=64, c2=64)
    sppf.cv1 = spp.cv1
    sppf.cv2 = spp.cv2
    r2 = sppf(x)
    print(r2.shape)

    print(torch.mean(torch.abs(r1 - r2)))


def t3():
    x = torch.rand(2, 256, 100, 100)
    down = DownSampling(c1=256, c2=256)
    r1 = down(x)
    print(r1.shape)
    up = UpSampling(c1=256, c2=256)
    r2 = up(r1)
    print(r2.shape)


if __name__ == '__main__':
    t3()
