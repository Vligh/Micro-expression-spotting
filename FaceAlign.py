# 作者：张鑫
# 时间：2022/8/22 15:05
#①使用dlib定位68个点
#②根据68个点选择面部切割框和鼻尖区域，记录68个点和切割框的相对位置
#③使用光流特征估计当前帧和初始帧之间的鼻尖运动(认为鼻尖运动为整体运动)
#④根据面部移动方向调整切割框，保证切割框和鼻尖区域相对位置保持不变
#⑤通过上一步的移动切割框，鼻尖区域相对位置也会随之移动，可能需要多次调整切割框，直到全局运动的振幅小于一个像素
#在确定切割框位置时，出现负值了，用不了

#自己的方法来对齐裁剪人脸：记录第一帧的鼻尖
import os
import dlib
import cv2

#找到68个点
faceDetector = dlib.get_frontal_face_detector()
landMarkPred = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

pos = []
pos2=[]
left_length,right_length=0,0
yr,xl,yl,xr=0, 0, 0, 0

#path = "15_0508funnydunkey_align1"
path = "1_0110pain_align"
files=os.listdir(path)
for file in files:
    if(file=='0.jpg'):
        img = cv2.imread(path + '/{}'.format(file))
        grayImg1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetector(grayImg1, 1)
        for face in faces:
            landMarks = landMarkPred(img, face)
            index = 0
            for point in landMarks.parts():
                pt_pos = (point.x, point.y)
                pos.append(pt_pos)
                #cv2.circle(img, pt_pos, 2, (0, 255), 1)
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(img, str(index), pt_pos, font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                index = index + 1

        '''
        #确定切割框的位置，鼻尖区域30，左眼角36，右眼角45
        #出现负值了，用不了
        xl = (pos[36][0]+pos[45][0])/2-4*(pos[36][0]-pos[45][0])/2
        xr = (pos[36][0]+pos[45][0])/2+4*(pos[36][0]-pos[45][0])/2
        yl = (pos[36][1]+pos[45][1])/2-3*(pos[36][1]+pos[45][1])/2
        yr = (pos[36][1]+pos[45][1])/2+5*(pos[36][1]+pos[45][1])/2
        pl=(xl,yl)#(638.0, -384.0)
        pr=(xr,yr)#(46.0, 1152.0)
        print(pl)
        print(pr)
        print("pos[36]",pos[36][0],pos[36][1])
        print("pos[45]",pos[45][0],pos[45][1])
        #出现一个问题，坐标有负值的
        cv2.rectangle(img1,pl,pr,(135,222,34),1)
        cv2.imshow('img', img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        # 鼻尖区域30，左眼角36，右眼角45，左边太阳穴0，右边16，下巴8，最上面的眉毛19和24
        xl = pos[0][0]   # 左边太阳穴的再往左5像素  -5
        yl = pos[19][1] -10  # 左边眉毛往上10像素   -10
        xr = pos[16][0]   #+5
        yr = pos[8][1]   #+5
        #cv2.rectangle(img, (xl, yl), (xr, yr), (135, 222, 34), 1)
        # 记录鼻尖与框的相对位置，鼻尖与左、下边框距离
        left_length = pos[30][0] - xl
        right_length = yr - pos[30][1]
        print(left_length, right_length)
        # 现在直到框的宽高和与鼻尖的相对位置
        # 切出来人脸区域
        out = img[yl:yr, xl:xr]
        cv2.imwrite("1_0110pain_crop/{}".format(file), out)
    if(file!='0.jpg'):
        pos2=[]
        img2 = cv2.imread(path + '/{}'.format(file))
        grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        faces = faceDetector(grayImg2, 1)
        for face in faces:
            landMarks = landMarkPred(img2, face)
            for point in landMarks.parts():
                pt_pos = (point.x, point.y)
                pos2.append(pt_pos)

        out2 = img2[pos2[30][1] - (pos[30][1] - yl):pos2[30][1] + right_length,
                    pos2[30][0] - left_length:pos2[30][0] + (xr - pos[30][0])]
        cv2.imwrite("1_0110pain_crop/{}".format(file), out2)

'''
#再对下一张进行处理，根据鼻尖坐标来调整框
faces2 = faceDetector(grayImg2,1)
pos2 = []
for face in faces2:
    landMarks = landMarkPred(img2,face)
    index = 0
    for point in landMarks.parts():
        pt_pos = (point.x,point.y)
        pos2.append(pt_pos)
        cv2.circle(img2,pt_pos,2,(0,255),1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img2,str(index),pt_pos,font,0.4,(0,255,255),1,cv2.LINE_AA)
        index = index+1

#调整框，确定框的位置
cv2.rectangle(img2,(pos2[30][0]-left_length,pos2[30][1]-(pos[30][1]-yl)),
              (pos2[30][0]+(xr-pos[30][0]),pos2[30][1]+right_length),(135,222,34),1)

#切割第二帧人脸
out2=img2_origin[pos2[30][1]-(pos[30][1]-yl):pos2[30][1]+right_length,
     pos2[30][0]-left_length:pos2[30][0]+(xr-pos[30][0])]
cv2.imwrite("15_0508funnydunkey_crop/img002.jpg",out2)
#现在有个问题，经过人脸旋转至“水平”后，可能人脸会歪。
'''
