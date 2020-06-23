# coding: utf-8

'''
CAM: Class Activation Map
gradient CAM
'''
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt 
import cv2



def grad_cam():
    tf.compat.v1.disable_eager_execution()   # K.gradients를 사용하기 위해....
    model = VGG16(weights='imagenet')  # C:\Users\BRAIN\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels.h5  540M
    
    img_path = './creative_commons_elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))  # PIL.Image.Image
    
    x = image.img_to_array(img)  # (224, 224, 3) 값응 정수이지만(0.0 ~ 255.0), float32
    
    plt.imshow(x/255.)
    
    plt.show()
    
    
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)  # -123.68 ~ 143.061   사이값으로 변횐되네....
    
    
    preds = model.predict(x)  # list return
    print(preds.shape)   #(1,1000)
    
    print('Predicted:', decode_predictions(preds, top=3)[0])  # C:\Users\BRAIN\.keras\models\imagenet_class_index.json   파일을 download받는다.
    print(np.argmax(preds[0]))
    
    
    
    # 예측 벡터의 '아프리카 코끼리' 항목
    african_elephant_output = model.output[:, 386]  # model.output: (N,1000) <----softmax 값.
    last_conv_layer = model.get_layer('block5_conv3')  # last_conv_layer.output: (N,14,14,512)
    
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
    
    
    pooled_grads = K.mean(grads,axis=(0,1,2)) 
    
    iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])
    
    
    pooled_grads_value, conv_layer_output_value = iterate(x)  # eager mode off인데도, numpy array x를 넣으면, value가 나온다. tensor는 아님.
    
    conv_layer_output_value *= pooled_grads_value
    
    
    heatmap = np.mean(conv_layer_output_value,axis=-1)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()
    
    
    # cv2 모듈을 사용해 원본 이미지를 로드합니다
    img = cv2.imread(img_path)
    
    # heatmap을 원본 이미지 크기에 맞게 변경합니다
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # heatmap을 RGB 포맷으로 변환합니다
    heatmap = np.uint8(255 * heatmap)
    
    # 히트맵으로 변환합니다
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 0.4는 히트맵의 강도입니다
    superimposed_img = heatmap * 0.4 + img
    plt.imshow(superimposed_img/255.)
    plt.show()
    
    cv2.imwrite('./elephant_cam.jpg', superimposed_img)
    cv2.imshow('original',superimposed_img/255.)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def saliency_map():
    tf.compat.v1.disable_eager_execution()   # K.gradients를 사용하기 위해....
    model = VGG16(weights='imagenet')  # C:\Users\BRAIN\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels.h5  540M
    
    img_path = './creative_commons_elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))  # PIL.Image.Image
    
    x = image.img_to_array(img)  # (224, 224, 3) 값응 정수이지만(0.0 ~ 255.0), float32
    
    plt.imshow(x/255.)
     
    plt.show()

    
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)  # -123.68~1.0사이값으로 변횐되네....
    
    
    preds = model.predict(x)  # list return
    print(preds.shape)   #(1,1000)
    
    print('Predicted:', decode_predictions(preds, top=3)[0])  # C:\Users\BRAIN\.keras\models\imagenet_class_index.json   파일을 download받는다.
    print(np.argmax(preds[0]))
    
    
    
    # 예측 벡터의 '아프리카 코끼리' 항목
    african_elephant_output = model.output[:, 386]  # model.output: (N,1000) <----softmax 값.
    last_conv_layer = model.get_layer('block5_conv3')  # last_conv_layer.output: (N,14,14,512)
    
    grads = K.max(K.abs(K.gradients(african_elephant_output, model.input)[0]),axis=-1)  # tf.gradients로 해도  OK
    
    
   
    
    iterate = K.function([model.input],[grads,last_conv_layer.output[0]])
    
    
    grads, conv_layer_output_value = iterate(x)  # eager mode off인데도, numpy array x를 넣으면, value가 나온다. tensor는 아님.
    
    saliency = grads[0]
    heatmap = saliency
    

    plt.figure(figsize=(12, 6))
    plt.subplot(1,3,1)

    plt.imshow(heatmap,interpolation="nearest")
    plt.title('imshow')
    
    plt.subplot(1,3,2)
    plt.imshow(heatmap,cmap=plt.cm.hot,interpolation="nearest")
    
    plt.title('imshow - cmap')
    
    plt.subplot(1,3,3)
    plt.imshow(img)
    
    plt.show()
    
def fooling():
    # imagenet class id: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    '''
    0: 'tench, Tinca tinca',1: 'goldfish, Carassius auratus',2: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
    3: 'tiger shark, Galeocerdo cuvieri',4: 'hammerhead, hammerhead shark',5: 'electric ray, crampfish, numbfish, torpedo',6: 'stingray',
    7: 'cock',8: 'hen',
    
    '''
    def make_fooling_image(X, target_y, model):
        """
        Generate a fooling image that is close to X, but that the model classifies
        as target_y.
    
        Inputs:
        - X: Input image, of shape (1, 224, 224, 3)
        - target_y: An integer in the range [0, 1000)
        - model: Pretrained SqueezeNet model
    
        Returns:
        - X_fooling: An image that is close to X, but that is classifed as target_y
        by the model.
        """
        X_fooling = X.copy()
        learning_rate = 1
        ##############################################################################
        # TODO: Generate a fooling image X_fooling that the model will classify as   #
        # the class target_y. Use gradient ascent on the target class score, using   #
        # the model.classifier Tensor to get the class scores for the model.image.   #
        # When computing an update step, first normalize the gradient:               #
        #   dX = learning_rate * g / ||g||_2                                         #
        #                                                                            #
        # You should write a training loop                                           #
        #                                                                            #  
        # HINT: For most examples, you should be able to generate a fooling image    #
        # in fewer than 100 iterations of gradient ascent.                           #
        # You can print your progress over iterations to check your algorithm.       #
        ##############################################################################


        for i in range(100):
            with tf.GradientTape() as tape:
                pred = model(X_fooling)
                scores = pred[0,target_y]
                
                #gradient = tf.gradients(scores,model.input)[0]
                #gradient = gradient / tf.norm(gradient)               
                
                
            classification_ = np.argmax(pred[0])
            if classification_ == target_y:
                break
        
            gradient = tf.gradients(scores,model.input)
            gradient = gradient / tf.norm(gradient)
            X_fooling += learning_rate * gradient_
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return X_fooling
    
    
    model = VGG16(weights='imagenet')
    img_path = './creative_commons_elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))  # PIL.Image.Image
    x = image.img_to_array(img)  # (224, 224, 3) 값응 정수이지만(0.0 ~ 255.0), float32
    

    
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)  # -123.68~1.0사이값으로 변횐되네....    (1, 224, 224, 3)
    target_y = 8 
    X_fooling = make_fooling_image(x, target_y, model)
    
    
    
    
if __name__ == "__main__":    
    #grad_cam()
    saliency_map()
    #fooling()
    print('Done')

