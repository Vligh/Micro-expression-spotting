# Micro-expression-spotting
Farneback 文件，其实是用的TVL1

在qq邮箱里有shape_predictor_68_face_landmarks.dat文件，2022.10.2这天：切割面部局部区域

shape_predictor_68_face_landmarks.dat文件是和这些py文件同级目录。
### Farneback文件

**TVL1.py**：是计算光流信息的，并且割开了眼睛区域。

**frameDifference.py**：是帧差特征，感觉不会这么简单（把图片相减）

**FeatureMap.py**：显示光流幅度随帧数改变

**test.py**：是计算Farneback算法的光流特征的

**Farneback.py**：不用这个

**FaceCrop.py**：把人脸区域切割出来并对齐 ，确定第一帧的鼻尖位置和框的大小，随后帧根据鼻尖位置来确定切割框位置（当时调着调着发现问题，没更新pos2的坐标导致后续帧切割框都跟随第一帧的框的坐标来切）所以没用这个切法

**FaceAlign.py**：把人脸区域切割出来并对齐用的是这个文件，记录第一帧的鼻尖区域的坐标和距离左边框下边框的距离，随后帧根据这些数据来调整框的位置进行切割。

**extractInfo.py**：这个文件就是试试手，看看TVL1算法的光流效果咋样

**ExtractFacePlus.py**：不能用

```python
#①想重新裁剪人脸区域，不直接用dlib.get_frontal_face_detector()里所检测的结果
#而是想：landmark 37和46 俩眼角之间的距离作为a,那么俩眼角连成线为l,l距离裁剪的上边框为0.3a
#l距离裁剪的下边框为1.1a，并且宽度为1.1a
#作废，裁剪出来的区域太小了
#②裁剪出来的像素不对不能用于光流计算
#ideal:①怎么保存像素大小一致的图片②怎么对齐人脸(摆正人脸还是歪的)
#③能不能以鼻子为基准点，每个图片的鼻子都是一个坐标来裁剪人脸
#④用眼角来摆正人脸不太行，试试用鼻子俩landmark来摆正
#现在情况是，裁剪出人脸的像素不一样，不能用于计算光流
```

**ExtractFace.py**：直接用dlib库进行切割人脸区域，用dlib给的边框来切，但是有很多非面部区域

**Detector68.py**：根据两个眼角连线，翻转至水平方向；（现在想在上面的基础上，（已经这样弄了））再根据鼻脊坐标27 30 之间的连线再进行翻转垂直操作

**Contrast.py**：这个不用，出错了。
