import cv2
import numpy as np

import os

# 读入图像
path_ori = r"E:\small defect Dataset\Test1\e1\e1.png"
filename = os.path.splitext(os.path.basename(path_ori))[0]
path_mask = os.path.join(os.path.dirname(path_ori), filename + "_mask.png")


img = cv2.imread(path_ori)

oriimg=img
# 将图像缩小为原来的四分之一
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

# 创建一个与图像大小相同的黑色图像
mask = np.zeros(img.shape[:2], np.uint8)

# 创建一个窗口来显示图像和mask
cv2.namedWindow("image")

# 鼠标回调函数，用于绘制mask
drawing = False  # 鼠标左键是否按下
points = []  # 点的列表


def draw_mask(event, x, y, flags, param):
    global mask, drawing, points

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
            img[mask == 255] = (0, 0, 0)
            # 绘制多边形
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, np.int32(points), 255)

            # 显示mask的轮廓
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(points) > 1:
            # 绘制多边形
            cv2.fillConvexPoly(mask, np.int32(points), 255)


# 设置鼠标回调函数
cv2.setMouseCallback("image", draw_mask)

while True:
    # 创建一个带有alpha通道的图像
    alpha = np.zeros(img.shape[:2], dtype=np.uint8)
    alpha[mask == 255] = 255
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = alpha

    # 显示图像和mask
    cv2.imshow("image", rgba)

    # 等待用户按键
    key = cv2.waitKey(1) & 0xFF

    # 如果按下q键，退出循环
    if key == ord("q"):
        break

# 将mask缩小为原来的四分之一

mask = cv2.resize(mask, (oriimg.shape[1], oriimg.shape[0]))

# 保存mask
cv2.imwrite(path_mask, mask)

# 关闭窗口
cv2.destroyAllWindows()
