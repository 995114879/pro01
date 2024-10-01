import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# 定义伽玛校正的函数
def adjust_gamma(image, gamma=1.0):
    # 创建一个查找表，范围从0到255
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # 使用查找表调整图像的伽玛值
    return cv.LUT(image, table)


def t1(img):
    # 顺时针旋转5度
    rows, cols, _ = img.shape
    M = cv.getRotationMatrix2D(center=(0, 0), angle=-5, scale=1)
    img = cv.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])

    # 截取
    img1 = np.zeros_like(img)
    img1[950:1260, 600:-600, :] = img[950:1260, 600:-600, :]

    # 转为灰度图
    img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    # 中值滤波
    img = cv.medianBlur(img, 5)

    # 腐蚀操作，将上方的横向白色区域腐蚀掉
    kernel = np.ones((5, 5), np.uint8)
    dst = cv.erode(img, kernel, iterations=1, borderType=cv.BORDER_REFLECT)

    # 设置伽玛值，小于1的值会提高对比度，大于1的值会降低对比度
    gamma = 0.3
    gamma_corrected = adjust_gamma(dst, gamma)

    # 二值化
    _, threshold = cv.threshold(gamma_corrected, 147, 255, cv.THRESH_TOZERO_INV)

    # 设置伽玛值，小于1的值会提高对比度，大于1的值会降低对比度
    gamma = 0.8
    gamma_corrected = adjust_gamma(threshold, gamma)

    # 二值化
    _, threshold = cv.threshold(gamma_corrected, 45, 255, cv.THRESH_BINARY)

    # 膨胀+腐蚀操作
    kernel = np.ones((3, 3), np.uint8)
    dst = cv.erode(threshold, kernel, iterations=1, borderType=cv.BORDER_REFLECT)
    dst = cv.dilate(dst, kernel, iterations=6)
    dst = cv.erode(dst, kernel, iterations=5, borderType=cv.BORDER_REFLECT)

    # Canny边缘检测
    blur = cv.GaussianBlur(dst, (5, 5), 0)
    edges = cv.Canny(blur, threshold1=50, threshold2=250)

    # 检测轮廓
    contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    contour = [len(t) for t in contours]
    print(len(contour))

    idx = np.argsort(contour)[16:17]
    cnts = [contours[i] for i in idx]

    img2 = np.zeros_like(img1)
    for i in range(len(cnts)):
        for x, y in cnts[i].reshape(cnts[i].shape[0], cnts[i].shape[2]):
            cv.circle(img2, (x, y), 5, (255, 255, 255), -1)


    # 逆时针旋转5度
    rows, cols, _ = img2.shape
    M = cv.getRotationMatrix2D(center=(0, 0), angle=5, scale=1)
    img2 = cv.warpAffine(img2, M, (cols, rows), borderValue=[0, 0, 0])



    cv.imwrite('./labels/new_label/Image_20240925113136473.jpg', img2)


if __name__ == '__main__':
    img = cv.imread("./datas/new_img/Image_20240925113136473.bmp")
    t1(img)
