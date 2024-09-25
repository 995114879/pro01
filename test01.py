import numpy as np
import cv2 as cv
"""
Image_20240814144256721.bmp
"""

# import matplotlib.pyplot as plt


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
    img1[400:-400, 1390:1660, :] = img[400:-400, 1390:1660, :]

    # 转为灰度图
    img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    # 中值滤波
    img = cv.medianBlur(img, 5)

    # 腐蚀操作，将上方的横向白色区域腐蚀掉
    kernel = np.ones((5, 5), np.uint8)
    dst = cv.erode(img, kernel, iterations=6, borderType=cv.BORDER_REFLECT)

    # 设置伽玛值，小于1的值会提高对比度，大于1的值会降低对比度
    gamma = 0.4
    gamma_corrected = adjust_gamma(dst, gamma)

    # 二值化
    _, threshold = cv.threshold(gamma_corrected, 10, 255, cv.THRESH_BINARY)

    # # 腐蚀+膨胀，去除下方大片白噪声
    # kernel = np.ones((5, 5), np.uint8)
    # dst1 = cv.erode(threshold, kernel, iterations=24, borderType=cv.BORDER_REFLECT)
    # kernel = np.ones((5, 5), np.uint8)
    # dst1 = cv.dilate(dst1, kernel, iterations=24)

    # Canny边缘检测
    blur = cv.GaussianBlur(threshold, (5, 5), 0)
    edges = cv.Canny(blur, threshold1=50, threshold2=250)

    # 检测轮廓
    contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # 获取最大轮廓
    idx = np.argmax([len(t) for t in contours])
    cnt = contours[idx]

    img2 = np.zeros_like(img1)
    for x, y in cnt.reshape(cnt.shape[0], cnt.shape[2]):
        if y > 990:  # 只保留下边
            cv.circle(img2, (x, y), 5, (255, 255, 255), -1)

    # 逆时针旋转5度
    rows, cols, _ = img2.shape
    M = cv.getRotationMatrix2D(center=(0, 0), angle=5, scale=1)
    img2 = cv.warpAffine(img2, M, (cols, rows), borderValue=[0, 0, 0])

    # 可视化
    # plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    # plt.title('img1')
    # plt.show()
    # cv.imshow('img1', img1)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 保存本地
    cv.imwrite('./labels/Image_20240814144256721.jpg', img2)


if __name__ == '__main__':
    img = cv.imread("./datas/Image_20240814144256721.bmp")
    t1(img)
