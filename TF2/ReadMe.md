## image classification from pretrained models
```
# model에 따라, image size를 맞춰야, 정확도가 나온다.
# 입력 image의 크기는 model.input.shape 으로 알아 낼 수 있다.
def preprocess_image(image_path,base_model,image_size):
    # 이미지를 로드하고, 모델이 정한 크기의 numpy array 로 변환
    img = image.load_img(image_path).resize(image_size)   # PIL.JpegImagePlugin.JpegImageFile image

    img = image.img_to_array(img)   # type은 float32 이지만, 값은 0~255값.
    img = np.expand_dims(img, axis=0)

    img = base_model.preprocess_input(img)  # 이미지  크기가 변하지 않는다.   ===> numpy array(tensor아님). 

    # return한 값의 범위는 모델에 따라 다르다.
    # inception: -1~1 사이값              resnet: -118.68 ~ 141.061 사이값(0~1사이값 아님)             vgg16: -118.68 ~ 141.061 사이값(0~1사이값 아님) 
    return img    

flag = 'resnet50'  # 'inception', 'resnet', 'vgg16'
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


```
## Learning Rate Decay with Callback
```
# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss='mse')
print(round(model.optimizer.lr.numpy(), 5))

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),epochs=15, callbacks=[callback], verbose=0)
print(round(model.optimizer.lr.numpy(), 5))

```
