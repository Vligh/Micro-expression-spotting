# 作者：张鑫
# 时间：2022/7/21 21:37
'''
calOpticalFlowFarneback() 参数
① prev：输入上一帧图像，为8位通道图
② next：输入当前帧图像，为8位通道图
③ flow：输出的光流矩阵，尺寸和输入图像一致，矩阵中每个元素都是一个Point2f类型的点，
        表示在输入图像中相同位置的像素点在上一帧的和当前帧图像中分别在x方向和y方向的位移，即(dx,dy)
④ pyr_scale：生成图像金字塔时上下两层的缩放比例，取值范围是0-1；当该参数为0.5时，即为经典的图像金字塔
⑤ level：生成的图像金字塔的层数；当level=0时表示不使用图像金字塔的FB稠密光流算法；一般取level=3；
⑥ winsize：表示滤波和检测的窗口大小，该参数越大对噪声抑制能力越强，并且能够检测快速移动目标（目标像素点不会移出窗口），
        但会引起运动区域的模糊；
⑦ iterations：对每层金字塔图像进行FB算法时的迭代次数；
⑧ poly_n：对当前像素点进行多项式展开时所选用的邻域大小，该参数值越大，运动区域模糊程度越大，
        对目标运动检测更稳定，会产生更鲁棒的算法和更模糊的运动场，官方推荐poly_n = 5或7；
⑨ poly_sigma：进行多项式展开时的高斯系数；推荐值为：当poly_n = 5时，poly_sigma = 1.1；
        当poly_n = 7时，poly_sigma = 1.5；
⑩ flag：进行光流估算的滤波器，有以下两种选择：
+ OPTFLOW_USE_INITIAL_FLOW 使用输入流作为初始流近似值，并使用盒子滤波器进行光流估算；
+ OPTFLOW_FARNEBACK_GAUSSIAN 使用高斯滤波器进行光流估算，高斯滤波器相比盒子滤波器的估算结果更精确，但运行速度较慢。
'''
import sys
import cv2
import numpy as np
import os

TVL1 = cv2.DualTVL1OpticalFlow_create()
fb_params = dict(pyr_scale = 0.5,levels = 3,winsize = 15,iterations = 3,
                 poly_n = 5,poly_sigma = 1.1,flags = 0)

path = "1_0102pain_crop"
files = os.listdir(path)
for file in files:
    if(file=='0.jpg'):
        print(0)
    elif(file!='0.jpg'):
        prev_initial = cv2.imread(path + '/0.jpg')
        prev_initial = cv2.resize(prev_initial, (268, 268))
        prev = cv2.cvtColor(prev_initial, cv2.COLOR_BGR2GRAY)

        next_initial = cv2.imread(path+"/{}".format(file))
        next_initial = cv2.resize(next_initial, (268, 268))
        next = cv2.cvtColor(next_initial, cv2.COLOR_BGR2GRAY)

        hsv = np.zeros_like(prev_initial)
        hsv[..., 2] = 255
        # hsv 0蕴含了相位信息，利用相位信息来计算色调，1表示饱和度，2表示亮度，通过振幅来计算亮度，振幅越大亮度越亮。
        flow = cv2.calcOpticalFlowFarneback(prev, next, None, **fb_params)
        # flow = TVL1.calc(prev, next, None)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # 找到了就是hsv[...,2]这里原来是1
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #cv2.imshow("Frame", bgr)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #
        cv2.imwrite("1_0102pain_farneback"+"/{}".format(file),bgr)



