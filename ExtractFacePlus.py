# 作者：张鑫
# 时间：2022/7/29 18:32
'''
①想重新裁剪人脸区域，不直接用dlib.get_frontal_face_detector()里所检测的结果
而是想：landmark 37和46 俩眼角之间的距离作为a,那么俩眼角连成线为l,l距离裁剪的上边框为0.3a
l距离裁剪的下边框为1.1a，并且宽度为1.1a
作废，裁剪出来的区域太小了
②裁剪出来的像素不对不能用于光流计算
ideal:①怎么保存像素大小一致的图片②怎么对齐人脸(摆正人脸还是歪的)
③能不能以鼻子为基准点，每个图片的鼻子都是一个坐标来裁剪人脸
④用眼角来摆正人脸不太行，试试用鼻子俩landmark来摆正
现在情况是，裁剪出人脸的像素不一样，不能用于计算光流
'''
import cv2
import dlib

img = cv2.imread("align_pic/8.jpg")
#得到landmark 37和46 的坐标，来裁剪人脸区域
faceDetector = dlib.get_frontal_face_detector()
landmarkPred = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
garyImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = faceDetector(garyImg,1)#得到脸部区域
origin_x,origin_y = 0,0
index = 0
for face in faces:
    landmarks = landmarkPred(img,face)
    bottom = landmarks.part(8).y#注意-1，landmark下标从0开始
    left = landmarks.part(17).x
    right = landmarks.part(26).x
    top = face.top()
    '''
    index = index + 1
    if(index == 0):
        origin_x = bottom - top
        origin_y = right - left
    '''
    print(top,bottom,left,right)

out = img[top:bottom,left:right]#y1:y2,x1:x2
cv2.resize(out,(128,128))
cv2.imshow("out",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("crop_face/8.jpg", out)