# 作者：张鑫
# 时间：2022/9/7 9:58
#一坨马赛克
import cv2
import numpy as np

img1 = cv2.imread("15_0508funnydunkey_crop/img001.jpg")
img2 = cv2.imread("15_0508funnydunkey_crop/img002.jpg")

img = img1 - img2

cv2.imshow("img",img)
cv2.waitKey(0)