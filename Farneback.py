# 作者：张鑫
# 时间：2022/7/21 21:16
#显示
import sys
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
'''
fb_params = dict(pyr_scale = 0.5,levels = 3,winsize = 15,iterations = 3,
                 poly_n = 5,poly_sigma = 1.2,flags = 0)

class app:
    def __init__(self, src):
        self.cap = cv.VideoCapture(src)

    def run(self):
        ret, preFrame = self.cap.read()
        if ret is not True:
            return
        preFrameGary = cv.cvtColor(preFrame, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(preFrame)
        hsv[..., 1] = 255

        while True:
            ret, curFrame = self.cap.read()
            if ret is not True:
                break
            curFrameGary = cv.cvtColor(curFrame, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(preFrameGary,curFrameGary, None, **fb_params)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imshow("Frame", bgr)

            ch = cv.waitKey(30) & 0xff
            if ch == 27:
                break

            preFrameGary = curFrameGary

        self.cap.release()

def main_func(argv):
    try:
        videoSrc = sys.argv[1]
    except:
        videoSrc = "vtest.avi"

    app(videoSrc).run()

if __name__ == '__main__':
    print(__doc__)
    main_func(sys.argv)
'''
speed=[32,111,138,28,59,77,97]
x=np.std(speed)
print("speed:",x)
#试试把图片的绿色通道的SD弄成柱状图，瞧瞧是否到微表情区间是否有起伏
data=[]
path = "1_0102pain_crop"
files = os.listdir(path)
for file in files:
    img = cv.imread(path+'/{}'.format(file))
    #print(np.std(img[...,1]))
    #print(np.mean(img[...,1]))
    data.append(np.mean(img[...,1]))

#归一化 Xnorm=(X-Xmin)/(Xmax-Xmin)
data_min = np.min(data)
data_max = np.max(data)
for i in range(0,len(data)):
    data[i] = (data[i]-data_min)/(data_max-data_min)
plt.plot(data)
plt.show()