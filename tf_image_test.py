# coding: utf-8

import skimage.io
from imageio import imread
import PIL
import numpy as np
import io
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2,json

def check_font():
    import matplotlib.font_manager as fm
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')  # 1172 ['C:\\Windows\\Fonts\\kartikab.ttf', 'C:\\Windows\\Fonts\\LSANSD.TTF', ...]
    font_list = fm.findSystemFonts(fontpaths=None, fontext='otf')  # 1172 ['C:\\Windows\\Fonts\\kartikab.ttf', 'C:\\Windows\\Fonts\\LSANSD.TTF', ...]
    
    A = [(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]  # [('NanumGothic', 'c:\\windows\\fonts\\nanumgothicbold.ttf'), ('NanumBarunGothic', 'c:\\windows\\fonts\\nanumbarungothic.ttf'), ...]
    
    print('현재 설정된  font:', plt.rcParams["font.family"])
    


def show_batch_images(batch_image,multi_flag=False):
    f = plt.figure()  # multi plt windows
    n = len(batch_image)
    for i in range(1,n+1):
        plt.subplot(1,n,i)
        plt.imshow(batch_image[i-1])
        plt.title('image{}'.format(i))   # plt.savefig('fig1.png', dpi=300)
    
    if multi_flag:
        plt.close(f)
        
        f.canvas.draw()
        image_from_plot = np.frombuffer(f.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(f.canvas.get_width_height()[::-1] + (3,))
        
        return image_from_plot
    else:
        return


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
        return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    
    
    img = tf.io.read_file(img_filename1)
    img = decode_img(img)
    
    print(img)
    
    sess = tf.Session()
    
    result = sess.run(img)  
    plt.imshow(result)
    plt.show()

def cv2_test():
    import cv2
    img_filename1 = 'D:/hccho/CommonDataset/VOCdevkit/VOC2007/train/JPEGImages/000019.jpg'

    img1 = skimage.io.imread(img_filename1)  # (375, 500, 3)  numpy array uint8  
    img2 = cv2.imread(img_filename1)  # RGB가 아니고, BGR로 읽는다. 3번째 channel에서 R <--> B가 바뀌어 있다. 값도 좀 다르다.
    img3 = cv2.resize(img2,(250,250))  # uint8
    print(img1.shape,img2.shape,img3.shape)
    show_batch_images([img1,img2,img3])
    plt.show()
    
    print(np.allclose(img1,img2))  # 
    
    img2 = img2[...,::-1].copy()
    show_batch_images([img1,img2])
    plt.show()
    
    print(np.allclose(img1,img2))  #     


def crop_test():
    import sys
    sys.path.append('D:/ObjectDetection/face-detection-ssd-hccho')
    
    import tf_extended as tfe
    def box_normalize(boxes,h,w):
        boxes_ = np.array(boxes)
        boxes_[:,:3:2] /= w
        boxes_[:,1:4:2] /= h
        
        boxes_.T[[0,1,2,3]] =  boxes_.T[[1,0,3,2]]
        
        return np.clip(boxes_,0.0,1.0)  # SSD 모델은 1보다 크면 error 난다.
    immage_filename = 'D:/hccho/CommonDataset/FDDB/images/2002_07_23_big_img_301.jpg'
    annotation_filename = 'D:/hccho/CommonDataset/FDDB/annotations/2002_07_23_big_img_301.json'
    
    image_originX = imread(immage_filename)
    image_originX = np.array(image_originX)
    h,w = image_originX.shape[:2]
    
    with open(annotation_filename, 'r') as f:
        anno = json.load(f)   
    for cls_name, boxes in anno.items():
        boxes = box_normalize(boxes,h,w)
    print('boxes: ', boxes)
    
    image = tf.convert_to_tensor(image_originX.astype(np.float32)/255.0)
    bboxes = tf.convert_to_tensor(boxes.astype(np.float32))
    
    # min_object_covered = 1로 하면 bounding box중 100% 이상 포함되는 것이 있다. 0.5이면 어떤 bounding box는 50%이상 포함된다.
    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, 0),
            min_object_covered=1,
            aspect_ratio_range=(0.6, 1.67),
            area_range=(0.1, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=False)

    # distort_bbox: 원본 이미지에서 crop해낸 부분
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),distort_bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    bboxes = tfe.bboxes_resize(distort_bbox[0, 0], bboxes)
    
    sess=tf.Session()
    
    multi_mode = False
    
    if multi_mode:
        matplotlib.use('agg')
        for i in range(100):
            img1,img2 = sess.run([image_with_box[0],distorted_image])
            result = show_batch_images([image_originX,img1,img2], multi_flag=True)
            plt.imsave("./result2/"+str(i)+".jpg",result)

    else:
        img1,img2 = sess.run([image_with_box[0],distorted_image])
        
        show_batch_images([image_originX,img1,img2])   

        plt.show()
    
    
    
    print('Done')

if __name__ == '__main__':
    #main_test()
    #resize_test()
    #cv2_test()
    crop_test()

    print('Done')
