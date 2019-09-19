# coding: utf-8

import skimage.io
import PIL
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

def show_batch_images(batch_image):
    plt.figure()  # multi plt windows
    n = len(batch_image)
    for i in range(1,n+1):
        plt.subplot(1,n,i)
        plt.imshow(batch_image[i-1])
        plt.title('image{}'.format(i))


img_filename1 = 'D:/hccho/CommonDataset/VOCdevkit/VOC2007/JPEGImages/000019.jpg'
img_filename2 = 'D:/hccho/CommonDataset/VOCdevkit/VOC2007/JPEGImages/000020.jpg'

# 방법 1
img1 = skimage.io.imread(img_filename1)  # (375, 500, 3)
img2 = skimage.io.imread(img_filename2)  # (500, 375, 3)


# 방법 2
with open(img_filename1, 'rb') as in_file:
    data = in_file.read()   # binary data  ---> 이런식으로 binary data로 만들어서 묶음으로 저장한다.


data = np.fromstring(data, dtype='uint8')  # binary --> int data(1d)
data2 = Image.open(io.BytesIO(data))  # <class 'PIL.JpegImagePlugin.JpegImageFile'>
print(data2.width, data2.height)
data3 = np.array(data2)  # (375, 500, 3)


#방법 3.
img11 = Image.open(img_filename1)  # (375, 500, 3)
img11 = img11.resize((300,300),Image.BICUBIC)

img22 = Image.open(img_filename2)  # (375, 500, 3)
img22 = img22.resize((300,300),Image.BICUBIC)

plt.subplot(1,2,1)
plt.imshow(img11)
plt.title('image1')

plt.subplot(1,2,2)
plt.imshow(img22)
plt.title('image2')
#plt.show()


# batch data
images = np.stack([img11,img22])/255  # (2,300,300,3)   0~1사이 값이어야 tensorflow가 이미지로 받아 들인다.
images = tf.convert_to_tensor(images)


boxes =np.array([ [0.0,0.0,0.5,0.5],[0.5,0.5,1,1],[0.25,0.25,0.75,1] ]).astype(np.float32)  # (y1,x1,y2,x2) 
boxes = tf.convert_to_tensor(boxes)

box_ind = np.array([1,0,1],dtype=np.int32)  # boxes 갯수와 일치해야 한다.
#box_ind = tf.convert_to_tensor(box_ind)

cropped_images = tf.image.crop_and_resize(images, boxes, box_ind, [100, 100])


sess = tf.Session()

result = sess.run(cropped_images)
show_batch_images(result)


plt.show()  # 마지막에 한번만 해야....multi plt windows





print('Done')