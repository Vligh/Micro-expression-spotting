# 作者：张鑫
# 时间：2022/8/9 16:41
#看SOFTNet中extraction_preprocess.py
#使用TVL1算法提取帧之间的光流特征
import dlib
import pandas as pd
import cv2
import numpy as np
import os
from FaceAlign import *
def pol2cart(rho,phi):#从极坐标转换为笛卡尔坐标，来计算光流应变
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x,y)

def computeStrain(u,v):
    u_x = u - pd.DataFrame(u).shift(-1,axis=1)
    v_y = v - pd.DataFrame(v).shift(-1,axis=0)
    u_y = u - pd.DataFrame(u).shift(-1,axis=0)
    v_x = v - pd.DataFrame(v).shift(-1,axis=1)
    os = np.array(np.sqrt(u_x**2 + v_y**2 + 1/2 * (u_y+v_x)**2).ffill(1).ffill(0))
    return os

#path = "1_0102pain_crop"
#path = "15_0508"
path = "15_0508funnydunkey_crop1"
files = os.listdir(path)
for file in files:
    if(file=='img001.jpg'):
        print(0)
    elif(file!='0.jpg'):
        img1 = cv2.imread(path+"/img001.jpg")
        #img1 = cv2.resize(img1, (268, 268))
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        img2 = cv2.imread(path+"/{}".format(file))
        #img2 = cv2.resize(img2, (268, 268))
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 计算光流特征
        optical_flow = cv2.DualTVL1OpticalFlow_create()
        # optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = optical_flow.calc(img1_gray, img2_gray, None)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        u, v = pol2cart(magnitude, angle)
        os = computeStrain(u, v)
        #cv2.imshow('optical strain', os)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #u = cv2.resize(u, (256, 256))
        #v = cv2.resize(v, (256, 256))
        #os = cv2.resize(os, (256, 256))

        # 特征聚合到256*256*3
        final = np.zeros((yr-yl, xr-xl, 3))
        final[:, :, 0] = u  # could not broadcast input array from shape (267,268) into shape (128,128)
        final[:, :, 1] = v
        final[:, :, 2] = os
        # 问题来了，图片的像素267*268，不能broadcast到128*128
        # 用resize调整大小了，但是不确定会不会有影响
        #cv2.imshow('final', final)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # 消除全局头部运动用鼻子区域
        x61, y61 = 0, 0  # nose landmark
        # 眼睛部分
        x11, y11, x12, y12, x13, y13, x14, y14, x15, y15, x16, y16 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        x21, y21, x22, y22, x23, y23, x24, y24, x25, y25, x26, y26 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        faceDetector = dlib.get_frontal_face_detector()
        landmarkpred = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        faces = faceDetector(img1_gray, 1)
        for face in faces:
            landmarks = landmarkpred(img1, face)
            # 鼻子
            x61 = landmarks.part(28).x
            y61 = landmarks.part(28).y
            # 左眼
            x11 = max(landmarks.part(36).x - 15, 0)
            y11 = landmarks.part(36).y
            x12 = landmarks.part(37).x
            y12 = max(landmarks.part(37).y - 15, 0)
            x13 = landmarks.part(38).x
            y13 = max(landmarks.part(38).y - 15, 0)
            x14 = min(landmarks.part(39).x + 15, yr-yl)  # 256是像素(resize)最大为256
            y14 = landmarks.part(39).y
            x15 = landmarks.part(40).x
            y15 = min(landmarks.part(40).y + 15, yr-yl)
            x16 = landmarks.part(41).x
            y16 = min(landmarks.part(41).y + 15, yr-yl)
            # 右眼
            x21 = max(landmarks.part(42).x - 15, 0)
            y21 = landmarks.part(42).y
            x22 = landmarks.part(43).x
            y22 = max(landmarks.part(43).y - 15, 0)
            x23 = landmarks.part(44).x
            y23 = max(landmarks.part(44).y - 15, 0)
            x24 = min(landmarks.part(45).x + 15, yr-yl)
            y24 = landmarks.part(45).y
            x25 = landmarks.part(46).x
            y25 = min(landmarks.part(46).y + 15, yr-yl)
            x26 = landmarks.part(47).x
            y26 = min(landmarks.part(47).y + 15, yr-yl)

        final[:, :, 0] = abs(final[:, :, 0] - final[y61 - 5:y61 + 6, x61 - 5:x61 + 6, 0].mean())
        final[:, :, 1] = abs(final[:, :, 1] - final[y61 - 5:y61 + 6, x61 - 5:x61 + 6, 1].mean())
        final[:, :, 2] = final[:, :, 2] - final[y61 - 5:y61 + 6, x61 - 5:x61 + 6, 2].mean()
        #cv2.imshow('remove movement final', final)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # 遮眼
        left_eye = [(x11, y11), (x12, y12), (x13, y13), (x14, y14), (x15, y15), (x16, y16)]
        right_eye = [(x21, y21), (x22, y22), (x23, y23), (x24, y24), (x25, y25), (x26, y26)]
        cv2.fillPoly(final, [np.array(left_eye)], 0)
        cv2.fillPoly(final, [np.array(right_eye)], 0)
        #cv2.imshow('eye mask final', final)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        temp = np.zeros_like(final)
        # temp[...,0] = final[...,0]
        # temp[...,1] = final[...,1]
        # temp[...,2] = final[...,2]
        temp = final
        #cv2.imshow('temp', temp)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite("15_0508funnydunkey_TVL11"+'/{}'.format(file) , temp * 255)  # 什么情况保存的图片是一片黑
        # https://blog.csdn.net/qq_37749442/article/details/101469161
        # 图片的数值应该在0-255，但是imwrite时已经被标准化了(设置在0-1之间)，只需要将标准化的值乘上255就行了


