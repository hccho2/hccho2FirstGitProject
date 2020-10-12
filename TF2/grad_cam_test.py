# coding: utf-8

'''
CAM: Class Activation Map
gradient CAM


tensorflow tutorial -- DeepDream: https://www.tensorflow.org/tutorials/generative/deepdream


C:\\Users\BRAIN\\.keras\\models   <-------------- pretrained file 다운로드   os.environ['USERPROFILE']  =  'C:\\Users\\BRAIN'
imagenet_class_index.json <----- 이것도 같이 다운받아진다.
'''
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import time

from tensorflow.keras.applications import inception_v3  # 입력이미지의 크기가 달라도 된다. 상태적인 크기로 변형된다.
from tensorflow.keras.applications import resnet


def deprocess_image(x):
    mean = [103.939, 116.779, 123.68]

    x[...,0] += mean[0]
    x[...,1] += mean[1]
    x[...,2] += mean[2]

    return x[...,::-1]


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
    # 주어진 image롤 class 확률에 대하여 미분하여, 미리 정한 다른 class로 분류되게 image를 update해 나간다.
    tf.compat.v1.disable_eager_execution()   # K.gradients를 사용하기 위해....
    
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
        learning_rate = 5

        score = model.output[:, target_y]
        
        grads = tf.gradients(score, model.input)[0]
        
        iterate = K.function([model.input],[model.output,grads])
        
        for i in range(100):
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
    target_y = 8 
    X_fooling = make_fooling_image(processed_x, target_y, model)
    
    preds = model.predict(X_fooling)  # list return
    fake_class = decode_predictions(preds, top=3)[0][0][1]
    print('Predicted:', decode_predictions(preds, top=3)[0])
    orig_img = x[0]    
    fool_img = deprocess_image(X_fooling[0])
    
    # Rescale 
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
    plt.title('Magnified difference (10x)')
    plt.imshow(deprocess_image(10 * (processed_x-X_fooling)[0])/255.)
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
        # foo loop 전체를 tf.function으로 묶을 수도 있지만, 여기서는 target_y와 같아지면, 멈춰야 하기 때문에 묶지 못한다.
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
    X_fooling = make_fooling_image3(processed_x, target_y, model).numpy()
    
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
    plt.title('Magnified difference (10x)')
    plt.imshow(deprocess_image(10 * (processed_x-X_fooling)[0])/255.)
    plt.axis('off')
    plt.gcf().tight_layout()
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
        return img
    
    
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
    layer_contributions = {
        'mixed2': 0.2,
        'mixed3': 3.,
        'mixed4': 2.,
        'mixed5': 1.5,
    }

    # 층 이름과 층 객체를 매핑한 딕셔너리를 만듭니다.
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    # 손실을 정의하고 각 층의 기여분을 이 스칼라 변수에 추가할 것입니다
    loss = K.variable(0.)
    for layer_name in layer_contributions:
        coeff = layer_contributions[layer_name]
        # 층의 출력을 얻습니다
        activation = layer_dict[layer_name].output
    
        scaling = K.prod(K.cast(K.shape(activation), 'float32'))
        # 층 특성의 L2 노름의 제곱을 손실에 추가합니다. 이미지 테두리는 제외하고 손실에 추가합니다.
        loss = loss + coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling


    # 이 텐서는 생성된 딥드림 이미지를 저장합니다
    dream = model.input
    
    # 손실에 대한 딥드림 이미지의 그래디언트를 계산합니다
    grads = K.gradients(loss, dream)[0]
    
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
    max_loss = 10.
    
    # 사용할 이미지 경로를 씁니다
    base_image_path = './original_photo_deep_dream.jpg'    # './original_photo_deep_dream.jpg'    './creative_commons_elephant.jpg'
    
    # 기본 이미지를 넘파이 배열로 로드합니다
    img = preprocess_image(base_image_path)
    
    # 경사 상승법을 실행할 스케일 크기를 정의한 튜플의 리스트를 준비합니다
    original_shape = img.shape[1:3]  # (350, 350)
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    
    # 이 리스트를 크기 순으로 뒤집습니다
    successive_shapes = successive_shapes[::-1]  # --->  [(178, 178), (250, 250), (350, 350)]
    
    # 이미지의 넘파이 배열을 가장 작은 스케일로 변경합니다
    original_img = np.copy(img)
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
    def preprocess_image(image_path,base_model,image_size):
        # 사진을 열고 크기를 줄이고 인셉션 V3가 인식하는 텐서 포맷으로 변환하는 유틸리티 함수
        img = image.load_img(image_path).resize(image_size)   # PIL.JpegImagePlugin.JpegImageFile image
        
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = base_model.preprocess_input(img)  # 이미지  크기가 변하지 않는다.
        return img    

    flag = 'inception'  # 'inception', 'resnet'
    if flag == 'inception':
        model = inception_v3.InceptionV3(weights='imagenet',include_top=True)  # include_top = True/False에 따라, weight 파일이 다르다.
        base_model = inception_v3
        image_size = (299,299)     # tuple(model.input.shape )[1:3]
    elif flag =='resnet':
        model = resnet.ResNet50(weights='imagenet',include_top=True)  # 100M 
        base_model = resnet
        image_size = (224,224)    # tuple(model.input.shape )[1:3]
    
    
    print(model.summary())
    base_image_path = './creative_commons_elephant.jpg'   # './original_photo_deep_dream.jpg'    './creative_commons_elephant.jpg'
    img = preprocess_image(base_image_path, base_model,image_size)
    
    print('preprocessed: ', img.shape)
    
    out = model(img)  # tensor
    preds = model.predict(img)  # np.array
    print(out.shape, np.argmax(out,axis=1))
    print(base_model.decode_predictions(preds, top=3))
    
    
    if flag == 'inception':
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        
        target_layers = [layer_dict['mixed10'].output,layer_dict['avg_pool'].output]
        
        model2 = K.function([model.input], target_layers)
        
        zzz = model2(img)
        print(zzz[0].shape, zzz[1].shape )
    elif flag == 'resnet':
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        
        target_layers = [layer_dict['conv5_block3_out'].output,layer_dict['avg_pool'].output]   
        model2 = K.function([model.input], target_layers)
        
        zzz = model2(img)
        print(zzz[0].shape, zzz[1].shape )    
    
if __name__ == "__main__":    
    #grad_cam()
    #saliency_map()
    #fooling()
    #fooling2()
    InceptionV3_Resnet_test()
    #deep_dream()
    
    print('Done')
