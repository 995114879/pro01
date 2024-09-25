import numpy as np
import cv2 as cv
"""
Image_20240814153033081.bmp
Image_20240814153036064.bmp
"""

# 定义伽玛校正的函数
def adjust_gamma(image, gamma=1.0):
    # 创建一个查找表，范围从0到255
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # 使用查找表调整图像的伽玛值
    return cv.LUT(image, table)


def t1(img):
    # 顺时针旋转13.5度
    rows, cols, _ = img.shape
    M = cv.getRotationMatrix2D(center=(0, 0), angle=-13.5, scale=1)
    img = cv.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])

    # 截取
    img1 = np.zeros_like(img)
    img1[300:-300, 1390:1660, :] = img[300:-300, 1390:1660, :]
    # 分别对每个颜色通道进行赋值
    img1[img1[:, :, 0] == 0, 0] = 255  # 红色通道
    img1[img1[:, :, 1] == 0, 1] = 255  # 绿色通道
    img1[img1[:, :, 2] == 0, 2] = 255  # 蓝色通道

    # 转为灰度图
    img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    # 中值滤波
    img = cv.medianBlur(img, 5)

    # 腐蚀操作
    kernel = np.ones((5, 5), np.uint8)
    dst = cv.erode(img, kernel, iterations=1, borderType=cv.BORDER_REFLECT)

    # 设置伽玛值，小于1的值会提高对比度，大于1的值会降低对比度
    gamma = 0.8
    gamma_corrected = adjust_gamma(dst, gamma)

    # 二值化
    _, threshold = cv.threshold(gamma_corrected, 100, 255, cv.THRESH_TOZERO_INV)

    kernel = np.ones((5, 5), np.uint8)
    dst1 = cv.dilate(threshold, kernel, iterations=2)

    # Canny边缘检测
    blur = cv.GaussianBlur(dst1, (5, 5), 0)
    edges = cv.Canny(blur, threshold1=50, threshold2=250)

    # 检测轮廓
    contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    contour = [len(t) for t in contours]
    idx = np.argsort(contour)[2:5]
    cnts = [contours[i] for i in idx]
    print(cnts[0].shape)

    img2 = np.zeros_like(img1)
    for i in range(len(cnts)):
        if i == 1:
            continue
        for x, y in cnts[i].reshape(cnts[i].shape[0], cnts[i].shape[2]):
            cv.circle(img2, (x, y), 5, (255, 255, 255), -1)

    # 逆时针旋转13.5度
    rows, cols, _ = img2.shape
    M = cv.getRotationMatrix2D(center=(0, 0), angle=13.5, scale=1)
    img2 = cv.warpAffine(img2, M, (cols, rows), borderValue=[0, 0, 0])

    cv.imwrite('labels/label/Image_20240814153036064.jpg', img2)


if __name__ == '__main__':
    img = cv.imread('datas/img/Image_20240814153036064.bmp')
    t1(img)
