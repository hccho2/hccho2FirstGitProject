# coding: utf-8

'''
CAM: Class Activation Map
gradient CAM


tensorflow tutorial -- DeepDream: https://www.tensorflow.org/tutorials/generative/deepdream


C:\\Users\BRAIN\\.keras\\models   <-------------- pretrained file 다운로드   os.environ['USERPROFILE']  =  'C:\\Users\\BRAIN'
imagenet_class_index.json <----- 이것도 같이 다운받아진다.




- include_top=True ---> Output Shape의 형태가 결정되어 있다.
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
conv1_pad (ZeroPadding2D)       (None, 230, 230, 3)  0           input_1[0][0]                    

....



- include_top=False
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
conv1_pad (ZeroPadding2D)       (None, 230, 230, 3)  0           input_2[0][0]                    
...


class_visualization:

아래 site는 코딩하는데는 참고하지 않았지만, 보면 좋을 듯.
https://timsainburg.com/tensorflow-2-feature-visualization-visualizing-classes.html   https://github.com/timsainb/tensorflow-2-feature-visualization-notebooks



'''
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import time,json
from scipy.ndimage.filters import gaussian_filter1d
from tensorflow.keras.applications import inception_v3  # 입력이미지의 크기가 달라도 된다. 상태적인 크기로 변형된다.
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications import vgg16

def deprocess_image(x):
    #  tensorflow의 VGG16의 preprocess_input의 되돌리는 함수가 없기 때문에, 직접 만들어야 한다.
    mean = [103.939, 116.779, 123.68]

    x[...,0] += mean[0]
    x[...,1] += mean[1]
    x[...,2] += mean[2]

    return np.clip(x[...,::-1],0,255).astype(np.uint8)


def preprocess_image(x):
    mean = [103.939, 116.779, 123.68]
    x = x[...,::-1]
    x[...,0] -= mean[0]
    x[...,1] -= mean[1]
    x[...,2] -= mean[2]

    return x


def grad_cam():
    tf.compat.v1.disable_eager_execution()   # K.gradients를 사용하기 위해....
    model = VGG16(weights='imagenet')  # C:\Users\BRAIN\.keras\models\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5  85M
    
    img_path = './creative_commons_elephant.jpg'  #(600,899)
    img = image.load_img(img_path, target_size=(224, 224))  # PIL.Image.Image
    
    x = image.img_to_array(img)  # (224, 224, 3) 값은 정수이지만(0.0 ~ 255.0), float32
    
    plt.imshow(x/255)  # 255로 나누어야 한다.
    
    plt.show()
    
    
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)  # -123.68 ~ 143.061   사이값으로 변횐되네....
    
    
    preds = model.predict(x)  # numpy array return shape(1,1000) --> 확률값. 합이 1
    print(preds.shape)   #(1,1000)
    
    print('Predicted:', decode_predictions(preds, top=3)[0])  # C:\Users\BRAIN\.keras\models\imagenet_class_index.json   파일을 download받는다.
    print(np.argmax(preds[0]))
    
    
    
    # 예측 벡터의 '아프리카 코끼리' 항목
    african_elephant_output = model.output[:, 386]  # model.output: (N,1000), tensor <----softmax 취한 확률 값.
    last_conv_layer = model.get_layer('block5_conv3')  # last_conv_layer.output: (N,14,14,512)
    
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
    
    
    pooled_grads = K.mean(grads,axis=(0,1,2)) 
    
    iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])
    
    
    pooled_grads_value, conv_layer_output_value = iterate(x)  # eager mode off인데도, numpy array x를 넣으면, value가 나온다. tensor는 아님.
    # pooled_grads_value(512,), conv_layer_output_value(14, 14, 512)

    conv_layer_output_value *= pooled_grads_value
    
    
    heatmap = np.mean(conv_layer_output_value,axis=-1)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()
    
    
    # cv2 모듈을 사용해 원본 이미지를 로드합니다
    img = cv2.imread(img_path)
    
    # heatmap을 원본 이미지 크기에 맞게 변경합니다
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  #(14,14) ==> (600,899)
    
    # heatmap을 RGB 포맷으로 변환합니다
    heatmap = np.uint8(255 * heatmap)
    
    # 히트맵으로 변환합니다
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # ---> (600, 899, 3)
    
    # 0.4는 히트맵의 강도입니다
    superimposed_img = heatmap * 0.4 + img
    plt.imshow(superimposed_img/255.)
    plt.show()
    
    cv2.imwrite('./elephant_cam.jpg', superimposed_img)
    cv2.imshow('original',superimposed_img/255.)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def saliency_map():
    # cs231n 숙제에서... D:\hccho\cs231n-Assignment\assignment3\NetworkVisualization-TensorFlow.ipynb
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
    
    saliency = grads[0]  # (224, 224)
    heatmap = saliency
    

    plt.figure(figsize=(12, 6))
    plt.subplot(1,3,1)

    plt.imshow(heatmap,interpolation="nearest",cmap='gray')
    plt.title('imshow')
    
    plt.subplot(1,3,2)
    plt.imshow(heatmap,cmap=plt.cm.hot,interpolation="nearest")
    
    plt.title('imshow - cmap(plt.cm.hot)')
    
    plt.subplot(1,3,3)
    plt.imshow(img)
    
    plt.show()
    
def fooling():
    # https://www.tensorflow.org/tutorials/generative/adversarial_fgsm
    # 주어진 image롤 class 확률에 대하여 미분하여, 미리 정한 다른 class로 분류되게 image를 update해 나간다.
    tf.compat.v1.disable_eager_execution()   # K.gradients를 사용하기 위해....
    
    # imagenet class id: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    '''
    0: 'tench, Tinca tinca',1: 'goldfish, Carassius auratus',2: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
    3: 'tiger shark, Galeocerdo cuvieri',4: 'hammerhead, hammerhead shark',5: 'electric ray, crampfish, numbfish, torpedo',6: 'stingray',
    7: 'cock',8: 'hen',
    
    '''
    def make_fooling_image(X, target_y, model):
        # cs231n 코드
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
        learning_rate = 5

        score = model.output[:, target_y]
        
        grads = tf.gradients(score, model.input)[0]
        
        iterate = K.function([model.input],[model.output,grads])
        
        for i in range(200):
            pred_,gradient_ = iterate(X_fooling)             
        
            classification_ = np.argmax(pred_[0])
            #print(i,classification_)
            if classification_ == target_y:
                break
            gradient_ = gradient_ / np.linalg.norm(gradient_)
            X_fooling += learning_rate * gradient_      

        return X_fooling
    
    
    model = VGG16(weights='imagenet')
    img_path = './creative_commons_elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))  # PIL.Image.Image
    x = image.img_to_array(img)  # (224, 224, 3) 값응 정수이지만(0.0 ~ 255.0), float32
    

    
    x = np.expand_dims(x,axis=0)
    processed_x = preprocess_input(x.copy())  # -123.68~1.0사이값으로 변횐되네....    (1, 224, 224, 3)
    target_y = 968 
    X_fooling = make_fooling_image(processed_x, target_y, model)
    
    preds = model.predict(X_fooling)  # list return
    fake_class = decode_predictions(preds, top=3)[0][0][1]    # 'hen'
    print('Predicted:', decode_predictions(preds, top=3)[0])
    orig_img = x[0]    
    fool_img = deprocess_image(X_fooling[0])
    
    # Rescale 
    plt.figure(figsize=(15,3))
    plt.subplot(1, 4, 1)
    plt.imshow(orig_img.astype(np.uint8))
    plt.axis('off')
    plt.title('African_elephant')
    plt.subplot(1, 4, 2)
    plt.imshow(fool_img/255.)
    plt.title(fake_class)
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.title('Difference')
    plt.imshow(deprocess_image((processed_x-X_fooling)[0]))
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.title('Magnified difference (100x)')
    mag_img = np.clip(100*deprocess_image((processed_x-X_fooling)[0]),0,255)
    plt.imshow(mag_img)
    plt.axis('off')
    plt.gcf().tight_layout()
    plt.show()
    
def fooling2():
    # fooling()과 달리 tf.GradientTape로 구현: 핵심은 tape.watch()
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
        s_time = time.time()
        X_fooling = tf.convert_to_tensor(X)
        learning_rate = 5

        for i in range(100):
            with tf.GradientTape() as tape:
                tape.watch(X_fooling)
                pred = model(X_fooling)
                score = pred[0, target_y]
            classification = tf.argmax(pred[0])
            if classification == target_y:
                break
            gradient = tape.gradient(score, X_fooling)
            gradient = gradient / tf.norm(gradient)
            X_fooling += learning_rate * gradient 
        
        
        print('elapsed: ', time.time()-s_time)
        return X_fooling
    


    def make_fooling_image2(X, target_y, model):
        # tf.function은 loading 시간이 있기 때문에, 빨라지지 않을 수도 있다. 여기서는 조금 빨라진다.
        # for loop 전체를 tf.function으로 묶을 수도 있지만, 여기서는 target_y와 같아지면, 멈춰야 하기 때문에 묶지 못한다. ---> make_fooling_image3에서 해결!!!
        # 
        s_time = time.time()
        X_fooling = tf.convert_to_tensor(X)
        learning_rate = 5

        @tf.function(input_signature=[tf.TensorSpec(shape=[1,224,224,3], dtype=tf.float32)])
        def train(X_fooling):
            with tf.GradientTape() as tape:
                tape.watch(X_fooling)
                pred = model(X_fooling)
                score = pred[0, target_y]
            gradient = tape.gradient(score, X_fooling)
            gradient = gradient / tf.norm(gradient)
            X_fooling += learning_rate * gradient             
            return tf.argmax(pred[0]), X_fooling
            
        
        
        c = np.argmax(model.predict(X_fooling),axis=-1)[0]
        for i in range(100):
            #print(i)
            if c== target_y:
                break
            c,X_fooling = train(X_fooling)
            c = c.numpy()
            
        print('elapsed: ', time.time()-s_time)
        return X_fooling

    def make_fooling_image3(X, target_y, model):
        # loop 전제를 tf.function으로.... y_target을 tain()의 argument로 넘기니까 되네...
        # 
        s_time = time.time()
        X_fooling = tf.convert_to_tensor(X)
        learning_rate = 5

        @tf.function(input_signature=[tf.TensorSpec(shape=[1,224,224,3], dtype=tf.float32),tf.TensorSpec(shape=[], dtype=tf.int32)])
        def train(X_fooling,y):
            y = tf.cast(y,tf.int32)
            for i in tf.range(100):
                with tf.GradientTape() as tape:
                    tape.watch(X_fooling)
                    pred = model(X_fooling)
                    score = pred[0, y]
                
                classification = tf.cast(tf.argmax(pred[0]),tf.int32)
                if classification == y:
                    break
                gradient = tape.gradient(score, X_fooling)
                gradient = gradient / tf.norm(gradient)
                X_fooling += learning_rate * gradient             
            return X_fooling
        
        #tf.config.experimental_run_functions_eagerly(True)
        X_fooling = train(X_fooling,target_y)
            
        print('elapsed: ', time.time()-s_time)
        return X_fooling



    model = VGG16(weights='imagenet')
    img_path = './creative_commons_elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))  # PIL.Image.Image
    x = image.img_to_array(img)  # (224, 224, 3) 값응 정수이지만(0.0 ~ 255.0), float32
    

    
    x = np.expand_dims(x,axis=0)
    processed_x = preprocess_input(x.copy())  # -123.68~1.0사이값으로 변횐되네....    (1, 224, 224, 3)
    target_y = 8 
    #X_fooling = make_fooling_image(processed_x, target_y, model).numpy()
    X_fooling = make_fooling_image2(processed_x, target_y, model).numpy()
    #X_fooling = make_fooling_image3(processed_x, target_y, model).numpy()
    
    preds = model.predict(X_fooling)  # list return
    fake_class = decode_predictions(preds, top=3)[0][0][1]
    print('Predicted:', decode_predictions(preds, top=3)[0])
    orig_img = x[0]    
    fool_img = deprocess_image(X_fooling[0])
    
    # Rescale 
    plt.figure(figsize=(15,3))
    plt.subplot(1, 4, 1)
    plt.imshow(orig_img/255.)
    plt.axis('off')
    plt.title('African_elephant')
    plt.subplot(1, 4, 2)
    plt.imshow(fool_img/225.)
    plt.title(fake_class)
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.title('Difference')
    plt.imshow(deprocess_image((processed_x-X_fooling)[0])/255.)
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.title('Magnified difference (100x)')
    mag_img = np.clip(100*deprocess_image((processed_x-X_fooling)[0]),0,255)
    plt.imshow(mag_img)
    plt.axis('off')
    plt.gcf().tight_layout()
    plt.show()





def class_visualization():
    '''
    cs231n assignment 코드 참조. assignment에서는 squeezenet을 사용. 나는 vgg16사용. processing된 image의 값의 범위가 달라 구현에 애를 먹음.
    1. vgg16의 softmax를 취한 값으로 하면 잘 안되고, logit을 이용해야 한다.
    2. learning rate도 중요하다. learning rate을 줄여주는 것이 아주 좋은 결과를 준다.
    
    
    '''
    

    class_idx = json.load(open(r"C:\Users\BRAIN\.keras\models\imagenet_class_index.json"))  # dict
    
    class_names = [class_idx[str(k)][1] for k in range(len(class_idx))]  # ['tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead', ...]
    def blur_image(X, sigma=1):
        X = gaussian_filter1d(X, sigma, axis=1)
        X = gaussian_filter1d(X, sigma, axis=2)
        return X

    def create_class_visualization(X, target_y, model, **kwargs):
        """
        Generate an image to maximize the score of target_y under a pretrained model.
        
        Inputs:
        - target_y: Integer in the range [0, 1000) giving the index of the class
        - model: A pretrained CNN that will be used to generate the image
        
        Keyword arguments:
        - l2_reg: Strength of L2 regularization on the image
        - learning_rate: How big of a step to take
        - num_iterations: How many iterations to use
        - blur_every: How often to blur the image as an implicit regularizer
        - max_jitter: How much to gjitter the image as an implicit regularizer
        - show_every: How often to show the intermediate result
        """
        mean = np.array([103.939, 116.779, 123.68])
        
        # kwargs로 넘거간 것이 없으면, default값으로...
        l2_reg = kwargs.pop('l2_reg', 1e-9)
        learning_rate = kwargs.pop('learning_rate', 500000)
        num_iterations = kwargs.pop('num_iterations', 1000)
        blur_every = kwargs.pop('blur_every', 10)
        max_jitter = kwargs.pop('max_jitter', 16)
        show_every = kwargs.pop('show_every', 500)
    
        #X = tf.convert_to_tensor(X)
        for t in range(num_iterations):
            if t>100:
                learning_rate = max(learning_rate*0.99,10000)
            # Randomly jitter the image a bit; this gives slightly nicer results
            ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
            X = tf.roll(tf.roll(X, ox, 1), oy, 2) # 좌우로 random하게 shift
            
            
            with tf.GradientTape() as tape:
                tape.watch(X)
                pred = model(X)
                loss = pred[0,target_y]  - l2_reg*tf.reduce_sum(tf.square(X))
                print(f'{t}: {class_names[tf.argmax(pred[0]).numpy()]} ----> {pred[0,target_y].numpy()}, lr = {learning_rate}')
            
            gradient_ = tape.gradient(loss,X)
            X += learning_rate * gradient_

    
            # Undo the jitter
            X = tf.roll(tf.roll(X, -ox, 1), -oy, 2)
    
            # As a regularizer, clip and periodically blur
            X = tf.clip_by_value(X, -mean, (255 - mean))
            if t % blur_every == 0:
                X = blur_image(X, sigma=0.5)
    
            # Periodically show the image
            if (t + 1) % show_every == 0 or t == num_iterations - 1:
                if tf.is_tensor(X):
                    X = X.numpy()
                plt.imshow(deprocess_image(X[0].copy()))
                class_name = class_names[target_y]
                plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
                plt.gcf().set_size_inches(4, 4)
                plt.axis('off')
                plt.show()
        return deprocess_image(X[0])


    #### classifier_activation=None으로 해야 결과가 좋다. 
    model = VGG16(weights='imagenet',classifier_activation=None)  # classifier_activation=None ---> softmax전 logit값이 return된다.

    x = np.random.randint(0,255,size=(224,224,3)).astype(np.float32)
    

    
    x = np.expand_dims(x,axis=0)
    processed_x = preprocess_input(x.copy())  # -123.68~1.0사이값으로 변횐되네....    (1, 224, 224, 3)
    target_y = 554  # 76: 'tarantula',

    X_generated = create_class_visualization(processed_x, target_y, model)

    plt.imsave(f'gen_{class_names[target_y]}.jpg',X_generated)
    plt.imshow(X_generated)
    plt.show()





def deep_dream():
    import scipy
    tf.compat.v1.disable_eager_execution()

    
    def eval_loss_and_grads(x):
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1]
        return loss_value, grad_values
    
    # 이 함수는 경사 상승법을 여러 번 반복하여 수행합니다
    def gradient_ascent(x, iterations, step, max_loss=None):
        # step = 0.01 --> learning rate
        for i in range(iterations):
            loss_value, grad_values = eval_loss_and_grads(x)
            if max_loss is not None and loss_value > max_loss:
                break
            print('...', i, '번째 손실 :', loss_value)
            x += step * grad_values
        return x

    def resize_img(img, size):
        img = np.copy(img)
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
        return scipy.ndimage.zoom(img, factors, order=1)
    
    
    def save_img(img, fname):
        pil_img = deprocess_image(np.copy(img))
        image.save_img(fname, pil_img)
    
    
    def preprocess_image(image_path):
        # 사진을 열고 크기를 줄이고 인셉션 V3가 인식하는 텐서 포맷으로 변환하는 유틸리티 함수
        img = image.load_img(image_path)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = inception_v3.preprocess_input(img)
        return img    # batch (N,H,W,C)
    
    
    def deprocess_image(x):
        # 넘파이 배열을 적절한 이미지 포맷으로 변환하는 유틸리티 함수
        if K.image_data_format() == 'channels_first':
            x = x.reshape((3, x.shape[2], x.shape[3]))
            x = x.transpose((1, 2, 0))
        else:
            # inception_v3.preprocess_input 함수에서 수행한 전처리 과정을 복원합니다
            x = x.reshape((x.shape[1], x.shape[2], 3))
        x /= 2.
        x += 0.5
        x *= 255.
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    #####################################
    #####################################


    K.set_learning_phase(0)
    model = inception_v3.InceptionV3(weights='imagenet',include_top=False)

    # model.summary()를 사용하면 모든 층 이름을 확인할 수 있습니다
    #print(model.summary())
    layer_contributions = {
        'mixed2': 0.2,
        'mixed3': 3.,
        'mixed4': 2.,
        'mixed5': 1.5,
    }


#     layer_contributions = {
#         'conv2d_52': 12,
#     }


    # 층 이름과 층 객체를 매핑한 딕셔너리를 만듭니다.
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    # 손실을 정의하고 각 층의 기여분을 이 스칼라 변수에 추가할 것입니다
    loss = K.variable(0.)
    for layer_name in layer_contributions.keys():
        coeff = layer_contributions[layer_name]
        # 층의 출력을 얻습니다
        activation = layer_dict[layer_name].output  # 'mixed2 shape: (None, None, None, 288)
    
        scaling = K.prod(K.cast(K.shape(activation), 'float32'))
        # 층 특성의 L2 노름의 제곱을 손실에 추가합니다. 이미지 테두리는 제외하고 손실에 추가합니다.
        loss = loss + coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling


    # 이 텐서는 생성된 딥드림 이미지를 저장합니다
    dream = model.input  # shape: (None, None, None, 3)
    
    # 손실에 대한 딥드림 이미지의 그래디언트를 계산합니다
    grads = K.gradients(loss, dream)[0]  # shape: (None, None, None, 3)
    
    # 그래디언트를 정규화합니다(이 기교가 중요합니다)
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  
    
    # 주어진 입력 이미지에서 손실과 그래디언트 값을 계산할 케라스 Function 객체를 만듭니다
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream], outputs)

    # 하이퍼파라미터를 바꾸면 새로운 효과가 만들어집니다
    step = 0.01  # 경상 상승법 단계 크기----> learning rate 역활
    num_octave = 3  # 경사 상승법을 실행할 스케일 단계 횟수
    octave_scale = 1.4  # 스케일 간의 크기 비율
    iterations = 20  # 스케일 단계마다 수행할 경사 상승법 횟수
    
    # 손실이 10보다 커지면 이상한 그림이 되는 것을 피하기 위해 경사 상승법 과정을 중지합니다
    max_loss = 100.
    
    # 사용할 이미지 경로를 씁니다
    base_image_path = './original_photo_deep_dream.jpg'    # './original_photo_deep_dream.jpg'    './creative_commons_elephant.jpg'  'mountain.jpg'
    
    # 기본 이미지를 넘파이 배열로 로드합니다
    img = preprocess_image(base_image_path)   # return batch (1,H,W,C)
    
    # 경사 상승법을 실행할 스케일 크기를 정의한 튜플의 리스트를 준비합니다
    original_shape = img.shape[1:3]  # (350, 350)
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    
    # 이 리스트를 크기 순으로 뒤집습니다
    successive_shapes = successive_shapes[::-1]  # --->  [(178, 178), (250, 250), (350, 350)]
    
    # 이미지의 넘파이 배열을 가장 작은 스케일로 변경합니다
    original_img = np.copy(img)   # (1, 350, 350, 3)
    shrunk_original_img = resize_img(img, successive_shapes[0])
    
    for shape in successive_shapes:  # num_octave만큼 반복
        print('처리할 이미지 크기', shape)
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              iterations=iterations,
                              step=step,
                              max_loss=max_loss)
        upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
        same_size_original = resize_img(original_img, shape)
        lost_detail = same_size_original - upscaled_shrunk_original_img
    
        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)
        save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
    
    save_img(img, fname='./final_dream.png')



    plt.imshow(plt.imread(base_image_path))
    plt.figure()
    
    plt.imshow(deprocess_image(np.copy(img)))
    plt.show()

def InceptionV3_Resnet_test():
    # model에 따라, image size를 맞춰야, 정확도가 나온다.
    # 입력 image의 크기는 model.input.shape 으로 알아 낼 수 있다.
    def preprocess_image(image_path,base_model,image_size):
        # 이미지를 로드하고, 모델이 정한 크기의 numpy array 로 변환
        img = image.load_img(image_path).resize(image_size)   # PIL.JpegImagePlugin.JpegImageFile image
        
        img = image.img_to_array(img)   # type은 float32 이지만, 값은 0~255값.
        img = np.expand_dims(img, axis=0)
        # inception_v3: img --> 2*(img/255-0.5)
        # resnet50: inception_v3와 전혀 다름.                                                  
        img = base_model.preprocess_input(img)  # 이미지  크기가 변하지 않는다.   ===> numpy array(tensor아님). 
        
        # return한 값의 범위는 모델에 따라 다르다.
        # inception: -1~1 사이값              resnet: -118.68 ~ 141.061 사이값(0~1사이값 아님)             vgg16: -118.68 ~ 141.061 사이값(0~1사이값 아님) 
        return img    

    flag = 'resnet50'  # 'inception', 'resnet', 'vgg16'     참고: classifier_activation=None으로 넘겨도 된다.
    if flag == 'inception_v3':
        model = inception_v3.InceptionV3(weights='imagenet',include_top=True)  # include_top = True/False에 따라, weight 파일이 다르다.
        base_model = inception_v3
        image_size = (299,299)     # tuple(model.input.shape )[1:3]
    elif flag =='resnet50':
        # weights: None(random initialization), 'imagenet'(pre-training on ImageNet), or the path to the weights file
        # include_top: whether to include the fully-connected layer at the top of the network.
        model = resnet.ResNet50(weights='imagenet',include_top=True)  # 100M, Trainable params: 25,583,592
        base_model = resnet
        image_size = (224,224)    # tuple(model.input.shape )[1:3]
    elif flag == 'vgg16':
        model = vgg16.VGG16(weights='imagenet',include_top=True)  # 100M 
        base_model = vgg16
        image_size = (224,224)    # tuple(model.input.shape )[1:3]        
    
    
    print(model.summary())
    base_image_path = './creative_commons_elephant.jpg'   # './original_photo_deep_dream.jpg'    './creative_commons_elephant.jpg'
    img = preprocess_image(base_image_path, base_model,image_size)  # numpy array로 변환
    
    print('preprocessed: ', img.shape)
    
    out = model(img)  # tensor  ---> softmax가 취해진 값
    preds = model.predict(img)  # np.array  ---> 값은 위의 out과 동일
    print('model: ',flag)
    print(out.shape, np.argmax(out,axis=1))
    print(base_model.decode_predictions(preds, top=3))  # preds로 부터 top 3를 추출해 준다.
    
    
    if flag == 'inception_v3':
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        
        target_layers = [layer_dict['mixed10'].output,layer_dict['avg_pool'].output]
        
        model2 = K.function([model.input], target_layers)
        
        zzz = model2(img)
        print(zzz[0].shape, zzz[1].shape )
    elif flag == 'resnet50':
        # layer dict를 만들어서 추룰할 수도 있고, model.get_layer를 사용할 수도 있다.
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        target_layers = [layer_dict['conv5_block3_out'].output,layer_dict['avg_pool'].output]   
        # layer_dict['conv5_block3_out'] 또는 model.get_layer('conv5_block3_out')
        model2 = K.function([model.input], target_layers)
        
        zzz = model2(img)
        print(zzz[0].shape, zzz[1].shape )    # (1, 7, 7, 2048) (1, 2048)


def model_functional_sequential():
    model = vgg16.VGG16(weights='imagenet',include_top=False)
    
    x = np.random.randn(3,224,224,3)
    
    y = model(x)
    print(y.shape)
    print('=='*10)


    # funcional api
    inputs = tf.keras.Input(shape=(224,224,3))
    vgg = vgg16.VGG16(weights='imagenet',include_top=False,input_tensor = inputs)

    print(vgg.get_layer('block3_pool'))
    model2 = tf.keras.Model(inputs,vgg.output)  # vgg.input과 inputs는 동일함.
    y2 = model2(x)
    print(y2.shape)


if __name__ == "__main__":    
    #grad_cam()
    #saliency_map()
    #fooling()
    #fooling2()
    #class_visualization()
    
    #InceptionV3_Resnet_test()
    #deep_dream()
    
    model_functional_sequential()
    
    print('Done')