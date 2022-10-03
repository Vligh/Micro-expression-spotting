# 作者：张鑫
# 时间：2022/8/19 16:18
#显示光流幅度随帧数改变
import os
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

data = []
#path = "1_0102pain_TVL1"
#path = "1_0102pain_farneback"
#path = "15_0508_TVL1"
path = "1_0110pain_TVL1"
files = os.listdir(path)
for file in files:
    img = cv2.imread(path+"/{}".format(file))
    temp = pow(np.mean(img[:, :, 0])**2+np.mean(img[:, :, 1])**2,0.5)
    #print(temp)
    data.append(temp)


#线性归一化 Xnorm=(X-Xmin)/(Xmax-Xmin)
data_min = np.min(data)
data_max = np.max(data)
for i in range(0,len(data)):
    data[i] = (data[i]-data_min)/(data_max-data_min)

#0均值标准化 z=(x-μ)/σ
#μ = np.mean(data)
#σ = np.var(data)
#for i in range(0,len(data)):
#    data[i] = (data[i]-μ)/σ

#效果都不太好,效果不好的意思是这个曲线看起来峰值差不多，没那么明显
plt.plot(data)
plt.xlabel('frame')
plt.ylabel('magnitude')
#plt.legend()
plt.show()