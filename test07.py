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
    # 截取
    img1 = np.zeros_like(img)
    img1[800:1140, 800:-500, :] = img[800:1140, 800:-500, :]

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
    gamma = 0.35
    gamma_corrected = adjust_gamma(threshold, gamma)

    # 二值化
    _, threshold = cv.threshold(gamma_corrected, 15, 255, cv.THRESH_BINARY)

    # 膨胀+腐蚀操作
    kernel = np.ones((5, 5), np.uint8)
    dst = cv.dilate(threshold, kernel, iterations=2)
    dst = cv.erode(dst, kernel, iterations=2, borderType=cv.BORDER_REFLECT)

    # Canny边缘检测
    blur = cv.GaussianBlur(dst, (5, 5), 0)
    edges = cv.Canny(blur, threshold1=50, threshold2=250)

    # 检测轮廓
    contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    contour = [len(t) for t in contours]

    idx = np.argsort(contour)[103:104]
    cnts = [contours[i] for i in idx]

    img2 = np.zeros_like(img1)
    for i in range(len(cnts)):
        for x, y in cnts[i].reshape(cnts[i].shape[0], cnts[i].shape[2]):
            cv.circle(img2, (x, y), 5, (255, 255, 255), -1)

    kernel = np.ones((5, 5), np.uint8)
    img2 = cv.dilate(img2, kernel, iterations=3)
    img2 = cv.erode(img2, kernel, iterations=2, borderType=cv.BORDER_REFLECT)

    cv.imwrite('./labels/new_label/Image_20240925113245769.jpg', img2)


if __name__ == '__main__':
    img = cv.imread("./datas/new_img/Image_20240925113245769.bmp")
    t1(img)
