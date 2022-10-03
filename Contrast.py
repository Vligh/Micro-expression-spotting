# 作者：张鑫
# 时间：2022/9/1 15:59
#不能用

# 随机调整对比度tf.image.random_contrast
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

image = tf.gfile.FastGFile("image/0.jpg", 'rb').read()
#image = cv2.imread("image/0.jpg")

with tf.compat.v1.Session() as sess:
     img_after_decode = tf.image.decode_png(image)

    # 函数原型random_contrast(image,lower,upper,seed)
    # 函数会在[lower upper]之间随机调整图像的对比度
    # 但要注意参数lower和upper都不能为负
    adjusted_contrast = tf.image.random_contrast(img_after_decode, 0.2, 18, )

    plt.imshow(adjusted_contrast.eval())
    plt.show()