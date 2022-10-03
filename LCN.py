# 作者：张鑫
# 时间：2022/8/20 15:42
#有错误没改
import cv2
import tensorflow as tf
import numpy as np
'''
print(tf.test.is_gpu_available())
tf.test.is_built_with_cuda()
tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)
'''
image = cv2.imread('crop_face/0.jpg')
init = tf.keras.initializers.constant(1/(2*9+1)**2)#r是邻域半径
layer = tf.keras.layers.Conv2D(filters=1,kernel_size=(2*9+1,2*9+1),
                               use_bias=False,kernel_initializer=init,
                               input_shape=image.shape)

image = image[:,:,np.newaxis]
mean = layer(image)#出错了 淦
square_mean = layer(image**2)
sigma = square_mean - mean**2
img_lcn = (image-mean)/np.sqrt(sigma)