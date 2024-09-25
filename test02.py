import numpy as np
import cv2 as cv

"""
Image_20240814152448912.bmp
Image_20240814152453512.bmp
"""
# import matplotlib.pyplot as plt


def t1(img):
    # 将图像从BGR转换为HSV颜色空间
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # 定义褐色的HSV范围
    # 范围可以根据实际图像中的褐色进行微调
    lower_brown = np.array([10, 100, 20])  # 褐色的下限
    upper_brown = np.array([20, 255, 200])  # 褐色的上限

    # 根据褐色范围创建掩膜
    mask = cv.inRange(hsv, lower_brown, upper_brown)

    # 创建一个空白图像，将符合条件的区域变成白色
    result = np.zeros_like(img)  # 初始化黑色图像
    result[mask != 0] = [255, 255, 255]  # 将掩膜中的部分变为白色

    # plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    # plt.title('Original')
    # plt.show()

    # 腐蚀+膨胀
    kernel = np.ones((5, 5), np.uint8)
    dst1 = cv.erode(result, kernel, iterations=1, borderType=cv.BORDER_REFLECT)
    kernel = np.ones((5, 5), np.uint8)
    dst1 = cv.dilate(dst1, kernel, iterations=2)

    # 保存本地
    cv.imwrite('./labels/Image_20240814152448912.jpg', dst1)


if __name__ == '__main__':
    img = cv.imread("./datas/Image_20240814152448912.bmp")
    t1(img)
