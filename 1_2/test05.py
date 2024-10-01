import numpy as np
import cv2 as cv
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
    # 将图片转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    (h, w) = img.shape[:2]
    redius = min(h // 2, w // 2) - 300
    center = (w // 2, h // 2)
    mask = np.zeros_like(gray)
    cv.circle(mask, center, redius, (255, 255, 255), -1)
    print(img.shape)
    print(mask.shape)
    gray = cv.bitwise_and(gray, gray, mask=mask)
    # cv.imwrite('img1.jpg', gray)
    # cv.imwrite('img.jpg', img)

    # 腐蚀操作
    kernel = np.ones((5, 5), np.uint8)
    gray = cv.erode(gray, kernel, iterations=1, borderType=cv.BORDER_REFLECT)

    # 使用Canny边缘检测
    edges = cv.Canny(gray, 150, 200, apertureSize=3)


    # 使用霍夫变换检测直线
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

    # 初始化角度列表
    angles = []

    # 遍历检测到的所有直线
    for line in lines:
        for rho, theta in line:
            # 转换theta为角度
            angle = np.degrees(theta)
            # 只考虑与水平线夹角小于30度和大于150度的直线
            # if angle < 50 or angle > 130:
            angles.append(angle)

    # 计算平均角度
    if angles:
        average_angle = np.mean(angles)
    else:
        average_angle = 0

    # 计算旋转矩阵
    print(angles)
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, average_angle, 1.0)

    # 执行旋转
    rotated = cv.warpAffine(gray, M, (w, h))




    # # 逆时针旋转13.5度
    # rows, cols, _ = img.shape
    # M = cv.getRotationMatrix2D(center=(0, 0), angle=13.5, scale=1)
    # img = cv.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])

    # # 截取
    # img1 = np.zeros_like(img)
    # img1[570:860, 820:-400, :] = img[570:860, 820:-400, :]

    # # 转为灰度图
    # img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    # 中值滤波
    img = cv.medianBlur(rotated, 5)

    # # 腐蚀操作
    # kernel = np.ones((5, 5), np.uint8)
    # dst = cv.erode(img, kernel, iterations=1, borderType=cv.BORDER_REFLECT)


    # 设置伽玛值，小于1的值会提高对比度，大于1的值会降低对比度
    gamma = 0.5
    gamma_corrected = adjust_gamma(img, gamma)



    # 二值化
    _, threshold = cv.threshold(gamma_corrected, 147, 255, cv.THRESH_TOZERO_INV)


    # 设置伽玛值，小于1的值会提高对比度，大于1的值会降低对比度
    gamma = 0.6
    gamma_corrected = adjust_gamma(threshold, gamma)


    # 二值化
    _, threshold = cv.threshold(gamma_corrected, 20, 255, cv.THRESH_BINARY)


    # 膨胀+腐蚀操作
    kernel = np.ones((5, 5), np.uint8)
    dst = cv.dilate(threshold, kernel, iterations=4)


    # Canny边缘检测
    blur = cv.GaussianBlur(dst, (5, 5), 0)
    edges = cv.Canny(blur, threshold1=50, threshold2=250)


    # 检测轮廓
    contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    contour = [len(t) for t in contours]
    # print(len(contour))

    idx = np.argsort(contour)[15:16] # [27,11,12,25]
    # cnts = [contours[25], contours[27]]
    cnts = [contours[i] for i in idx]

    img2 = np.zeros_like(img)
    for i in range(len(cnts)):
        for x, y in cnts[i].reshape(cnts[i].shape[0], cnts[i].shape[2]):
            cv.circle(img2, (x, y), 10, (255, 255, 255), -1)



    kernel = np.ones((5, 5), np.uint8)
    img2 = cv.dilate(img2, kernel, iterations=2)
    img2 = cv.erode(img2, kernel, iterations=2, borderType=cv.BORDER_REFLECT)


    # # 顺时针旋转13.5度
    # rows, cols, _ = img2.shape
    # M = cv.getRotationMatrix2D(center=(0, 0), angle=-13.5, scale=1)
    # img2 = cv.warpAffine(img2, M, (cols, rows), borderValue=[0, 0, 0])

    M = cv.getRotationMatrix2D(center, -average_angle, 1.0)

    # 执行旋转
    img2 = cv.warpAffine(img2, M, (w, h))

    cv.imwrite('../labels/new_label/Image_20240925112858851.jpg', img2)


if __name__ == '__main__':
    img = cv.imread("../datas/new_img/Image_20240925112858851.bmp")
    t1(img)
