# 作者：张鑫
# 时间：2022/9/18 9:27
#由于FaceAlign切割的人脸框肉眼看起来都不齐，所以再试试新的确定切割框位置的方法
#目前这样想：在人脸对齐(水平)的图像上，确定第一帧的鼻尖位置和框的大小，随后帧根据鼻尖位置来确定切割框位置
import os
import dlib
import cv2

#找到68个点
faceDetector = dlib.get_frontal_face_detector()
landMarkPred = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

pos = []
pos2=[]
box_wide,box_hight = 0,0
left_length,right_length=0,0
yr,xl,yl,xr=0, 0, 0, 0

path = "15_0508funnydunkey_align1"
files=os.listdir(path)

for file in files:
    if(file=='img001.jpg'):
        img = cv2.imread(path + '/{}'.format(file))
        grayImg1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetector(grayImg1, 1)
        for face in faces:
            landMarks = landMarkPred(img, face)
            for point in landMarks.parts():
                pt_pos = (point.x, point.y)
                pos.append(pt_pos)

        Lx = pos[0][0]  # 左上角的x坐标对应于landmark为0
        Rx = pos[16][0]  # 右下角x左边对应于landmark为16
        Ly = pos[19][1] - 10  # 左上角y对应于眉毛19再往上10像素
        Ry = pos[8][1] + 5  # 右下角y对应于8
        # 记录box的大小
        box_wide = Rx - Lx
        box_hight = Ry - Ly
        out = img[Ly:Ry,Lx:Rx]
        cv2.imwrite("15_0508funnydunkey_crop2/{}".format(file),out)

    if(file!='img001.jpg'):
        pos2=[]
        img2 = cv2.imread(path + '/{}'.format(file))
        grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        faces2 = faceDetector(grayImg2, 1)
        for face in faces2:
            landMarks = landMarkPred(img2, face)
            for point in landMarks.parts():
                pt_pos = (point.x, point.y)
                pos2.append(pt_pos)

        Lx = pos2[0][0]  # 左上角的x坐标对应于landmark为0
        Rx = pos2[16][0]  # 右下角x左边对应于landmark为16
        Ly = pos2[19][1] - 10  # 左上角y对应于眉毛19再往上10像素
        Ry = pos2[8][1] + 5  # 右下角y对应于8
        box1_wide = Rx - Lx
        box1_hight = Ry - Ly
        if box1_hight != box_hight:
            if box1_hight > box_hight:
                Ly += box1_hight - box_hight
            else:
                Ly -= box_hight - box1_hight
        if box1_wide != box_wide:
            if box1_wide > box_wide:
                Lx -= (box1_wide - box_wide) / 2
                Rx += (box1_wide - box_wide)-(box1_wide - box_wide) / 2
            else:
                Lx += (box_wide - box1_wide) / 2
                Rx += (box1_wide - box_wide)-(box1_wide - box_wide) / 2
        print(Lx,Ly,Rx,Ry)
        out1 = img2[Ly:Ry, Lx:Rx]
        cv2.imwrite("15_0508funnydunkey_crop2/{}".format(file), out1)

#有问题，对单个177图像进行框定位很对很正，但是到批处理那块就不行了
#这个在就算Ly Lx 时用的第一帧的pos，没用pos2的新的后续帧的landmarks
"""
#下面是对单个图像的确定框的实验，把最后的“”“放到上面for循环可以切换为批处理
img = cv2.imread("15_0508funnydunkey_align1/img177.jpg")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceDetector(grayImg, 1)
for face in faces:
    landMarks = landMarkPred(img, face)
    index = 0
    for point in landMarks.parts():
        pt_pos = (point.x, point.y)
        pos.append(pt_pos)
        # cv2.circle(img, pt_pos, 2, (0, 255), 1)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(index), pt_pos, font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        index = index + 1

#俩内眼角 39 42
Lx = pos[0][0]#左上角的x坐标对应于landmark为0
Rx = pos[16][0]#右下角x左边对应于landmark为16
Ly = pos[19][1]-10#左上角y对应于眉毛19再往上10像素
Ry = pos[8][1]+5#右下角y对应于8
#记录box的大小
box_wide = Rx-Lx
box_hight = Ry-Ly
print(box_wide,box_hight)
#记录鼻尖相对位置
nose_left = pos[30][0]-Lx
nose_top = pos[30][1]-Ly
print(nose_top,nose_left)
cv2.rectangle(img,(Lx,Ly),(Rx,Ry),(135,222,34),1)
cv2.imshow('img', img)
cv2.waitKey(0)
out = img[Ly:Ry,Lx:Rx]
cv2.imshow('out', out)
cv2.waitKey(0)

#鼻尖位置30  可以根据鼻尖与切割框的左上角的相对位置来调整


#想到：得到起始帧的框的大小，后续帧直接进行框的定位，根据起始帧的框的大小来调整后续帧的框的大小
img1 = cv2.imread("15_0508funnydunkey_align1/img092.jpg")
grayImg = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
faces = faceDetector(grayImg, 1)
for face in faces:
    landMarks = landMarkPred(img1, face)
    index = 0
    for point in landMarks.parts():
        pt_pos = (point.x, point.y)
        pos.append(pt_pos)
        index = index + 1

Lx = pos[0][0]#左上角的x坐标对应于landmark为0
Rx = pos[16][0]#右下角x左边对应于landmark为16
Ly = pos[19][1]-10#左上角y对应于眉毛19再往上10像素
Ry = pos[8][1]+5#右下角y对应于8

box1_wide = Rx-Lx
box1_hight = Ry-Ly
if box1_hight!=box_hight:
    if box1_hight>box_hight:
        Ly += box1_hight-box_hight
    else:
        Ly -= box_hight-box1_hight
if box1_wide!=box_wide:
    if box1_wide>box_wide:
        Lx -= (box1_wide-box_wide)/2
        Rx += (box1_wide-box_wide)/2
    else:
        Lx += (box_wide-box1_wide)/2
        Rx += (box1_wide-box_wide)/2
out1 = img1[Ly:Ry,Lx:Rx]
cv2.imshow('out1', out1)
cv2.waitKey(0)
box1_wide = Rx-Lx
box1_hight = Ry-Ly
print(box1_wide,box1_hight)
"""



