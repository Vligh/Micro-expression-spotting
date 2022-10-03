# 作者：张鑫
# 时间：2022/7/24 10:29
#裁剪出人脸区域用  可以看看下面两篇文章
#https://zhuanlan.zhihu.com/p/104299394
#https://blog.csdn.net/weixin_43854960/article/details/103925120
import cv2
import dlib
import os
import sys

#path = "1_0102pain_align"
path = "2_0101pain_align"
files = os.listdir(path)
for file in files:
    img = cv2.imread(path+'/{}'.format(file))
    faceDetector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    rect = faceDetector(img, 1)
    print(rect)  # rectangles[[(22, 82) (290, 350)]]'dlib.rectangle'没有下标
    print(rect[0])  # [(22, 82) (290, 350)]
    # 有问题'dlib.rectangle'类型的数据怎么转换为其他类型的数据

    x1 = rect[0].left()  # 22
    y1 = rect[0].top()  # 82
    x2 = rect[0].right()  # 290
    y2 = rect[0].bottom()  # 350
    # print(x1,y1,x2,y2)#22 82 290 350

    out = img[y1:y2, x1:x2]  # y1:y2,x1:x2
    out = cv2.resize(out,(128,128))
    cv2.imwrite('2_0101pain_crop'+'/{}'.format(file), out)
    #cv2.imshow("out", out)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # 接着就进行人脸对齐，把人脸弄水平
    # 想先对图片对人脸调整至水平，在切割出来，在Detector68.py文件下进行人脸对齐

