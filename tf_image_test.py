# coding: utf-8

import skimage.io
from imageio import imread
import PIL
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
def check_font():
    import matplotlib.font_manager as fm
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')  # 1172 ['C:\\Windows\\Fonts\\kartikab.ttf', 'C:\\Windows\\Fonts\\LSANSD.TTF', ...]
    font_list = fm.findSystemFonts(fontpaths=None, fontext='otf')  # 1172 ['C:\\Windows\\Fonts\\kartikab.ttf', 'C:\\Windows\\Fonts\\LSANSD.TTF', ...]
    
    A = [(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]  # [('NanumGothic', 'c:\\windows\\fonts\\nanumgothicbold.ttf'), ('NanumBarunGothic', 'c:\\windows\\fonts\\nanumbarungothic.ttf'), ...]
    
    print('현재 설정된  font:', plt.rcParams["font.family"])
    


def show_batch_images(batch_image):
    plt.figure()  # multi plt windows
    n = len(batch_image)
    for i in range(1,n+1):
        plt.subplot(1,n,i)
        plt.imshow(batch_image[i-1])
        plt.title('image{}'.format(i))



def cv2_rectangle_test():
    
    img1 = skimage.io.imread(img_filename1)
    cv2.rectangle(img1,(0,50),(50,60),color=(255, 255, 0), thickness=2) # (x1,y1),(x2,y2)
    plt.imshow(img1)
    plt.show()

def main_test():
    check_font()
    img_filename1 = 'D:/hccho/CommonDataset/VOCdevkit/VOC2007/JPEGImages/000019.jpg'
    img_filename2 = 'D:/hccho/CommonDataset/VOCdevkit/VOC2007/JPEGImages/000020.jpg'
    
    # 방법 1
    img1 = skimage.io.imread(img_filename1)  # (375, 500, 3)  numpy array uint8
    img2 = skimage.io.imread(img_filename2)  # (500, 375, 3)
    
    
    # 방법 2
    with open(img_filename1, 'rb') as in_file:
        data = in_file.read()   # binary data  ---> 이런식으로 binary data로 만들어서 묶음으로 저장한다.
    
    
    data = np.fromstring(data, dtype='uint8')  # binary --> int data(1d)
    data2 = Image.open(io.BytesIO(data))  # <class 'PIL.JpegImagePlugin.JpegImageFile'>
    print(data2.width, data2.height)
    data3 = np.array(data2)  # (375, 500, 3)
    print(np.allclose(img1, data3))
    
    
    # 또 다른 방법
    data4 = imread(img_filename1)
    print(np.allclose(img1, data4))
    
    
    #방법 3.
    img11 = Image.open(img_filename1)  # (375, 500, 3)
    img11 = img11.resize((300,300),Image.BICUBIC)
    
    img22 = Image.open(img_filename2)  # (375, 500, 3)
    img22 = img22.resize((300,300),Image.BICUBIC)
    
    
    
    plt.rcParams["font.family"] = 'NanumGothic'
    plt.subplot(1,2,1)
    plt.imshow(img11)
    plt.title('image1 한글')
    
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

def resize_test():
    img_filename1 = 'D:/hccho/CommonDataset/VOCdevkit/VOC2007/JPEGImages/000019.jpg'
    IMG_WIDTH = 300
    IMG_HEIGHT = 300
    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    
    
    img = tf.io.read_file(img_filename1)
    img = decode_img(img)
    
    print(img)
    
    sess = tf.Session()
    
    result = sess.run(img)  
    plt.imshow(result)
    plt.show()
if __name__ == '__main__':
    #main_test()
    resize_test()

    print('Done')