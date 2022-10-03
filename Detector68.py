# 作者：张鑫
# 时间：2022/7/23 17:40
#人脸对齐
#根据两个眼角连线，翻转至水平方向
#现在想在上面的基础上再根据鼻脊坐标27 30 之间的连线再进行翻转垂直操作
import math
import cv2
import dlib
import os

#path = "1_0102pain"
path = "1_0110pain"
files = os.listdir(path)
for file in files:
    img_origin = cv2.imread(path+'/{}'.format(file))
    img = cv2.imread(path+'/{}'.format(file))
    print(path+'/{}'.format(file))

    garyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 得到face detectors
    faceDetector = dlib.get_frontal_face_detector()
    # 得到landmark detectors
    landmarkPred = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    eye_x, eye_y = (0, 0), (0, 0)
    nose_x, nose_y = (0, 0), (0, 0)

    faces = faceDetector(garyImg, 1)  # 这个faceDetetor可能检测出多个人脸
    for face in faces:
        # print(face)#[(22, 82) (290, 350)]  这三行说明了face包含人脸区域的左上角和右下角坐标
        cv2.circle(img, (faces[0].left(), faces[0].top()), 2, (0, 0, 255), -1)
        cv2.circle(img, (faces[0].right(), faces[0].bottom()), 2, (0, 0, 255), -1)
        # 得到每个脸的68个地标点
        landmarks = landmarkPred(img, face)
        #print("landmark=37.x:", landmarks.part(37).x)
        index = 0
        for point in landmarks.parts():
            # print(point)#(4,150)...是一些坐标
            pt_pos = (point.x, point.y)  # 这里要转换一下，下面的circle和putText才能用
            if (index == 36):
                eye_x = pt_pos
            if (index == 45):
                eye_y = pt_pos
            if (index == 27):
                nose_x = pt_pos
            if (index == 30):
                nose_y = pt_pos
            cv2.circle(img, pt_pos, 2, (0, 255), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(index), pt_pos, font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            index = index + 1

    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    '''
    eye_center = ((landmarks[36,0]+landmarks[45,0])*1./2,
                  (landmarks[45,0]+landmarks[45,1])*1./2)
    #'dlib.full_object_detection' object is not subscriptable
    print(eye_center)
    得到两眼角的坐标 对应landmark为36和45
    '''
    print("The coordinates of the corners of the eyes:", eye_x, eye_y)  # (64, 166) (238, 163)
    eye_center = ((eye_x[0] + eye_y[0]) * 1. / 2, (eye_x[1] + eye_y[1]) * 1. / 2)
    print("eye_center:", eye_center)  # (151.0, 164.5)
    dx = eye_y[0] - eye_x[0]
    dy = eye_y[1] - eye_x[1]
    angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, 1)  # 计算仿射矩阵
    align_face = cv2.warpAffine(img_origin, RotateMatrix, (img_origin.shape[1], img_origin.shape[0]))  # 旋转
    # warpAffine函数最后一个参数，表示输出图像的尺寸，第一次用的(img.shape[0],img.shape[1])图像扁了
    #cv2.imshow('img', align_face)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # 先注释掉下面保存图片
    #cv2.imwrite(path+"_align"+'/{}'.format(file),align_face)
    #print(path+"_align"+'/{}'.format(file))

    #根据鼻脊连线再次进行对齐 27 30
    nose_center = ((nose_x[0] + nose_y[0]) * 1. / 2, (nose_x[1] + nose_y[1]) * 1. / 2)
    nose_dx = nose_y[0] - nose_x[0]
    nose_dy = nose_y[1] - nose_x[1]
    nose_angle = math.atan2(nose_dy,nose_dx)*1./math.pi
    RotateMatrix = cv2.getRotationMatrix2D(nose_center, nose_angle, 1)  # 计算仿射矩阵
    align_face = cv2.warpAffine(align_face, RotateMatrix, (align_face.shape[1], align_face.shape[0]))
    #cv2.imshow('img1', align_face)
    #cv2.waitKey(0)

    cv2.imwrite(path + "_align" + '/{}'.format(file), align_face)

