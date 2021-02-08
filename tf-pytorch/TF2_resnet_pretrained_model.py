
'''
dog cat 품종 분류 문제를 tf2로 구현.


'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications import resnet
import os

print('tf version: ', tf.__version__)


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """
    fig, axs = plt.subplots(1,2,figsize=(12, 5))
    
    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()



data_dir = r'D:\hccho\CommonDataset\Pet-Dataset-Oxford\images'

mode = 1
if mode==1:
    # weights='imagenet' 으로 하면 train이 잘 된다. initial_learning_rate=0.01  10 epoch train하면 valid acc = 85%
    # weights= None이면 train이 느리다.
    
    pretrained = 'imagenet'
    rescale = 1.0
    preprocessing_function = resnet.preprocess_input
else:
    # train이 느리게 된다.
    pretrained = None
    rescale = 1/255.
    preprocessing_function = None

train_datagen = ImageDataGenerator(rescale=rescale, shear_range=0.1, zoom_range=0.2,height_shift_range=0.2, 
                                   horizontal_flip=True,validation_split=0.2,
                                   preprocessing_function=preprocessing_function)


batch_size = 64
train_generator = train_datagen.flow_from_directory(data_dir, target_size=(224, 224), batch_size=batch_size,
                                                    class_mode='sparse',shuffle=True,subset='training')

valid_generator = train_datagen.flow_from_directory(data_dir, target_size=(224, 224), batch_size=batch_size,
                                                    class_mode='sparse',shuffle=False,subset='validation')


print('class name: ', train_generator.class_indices)  # list(train_generator.class_indices) ==> ['ants', 'bees']
print('batch_size = ',  train_generator.batch_size, 'image shape: ', train_generator.image_shape)
n_class = len(train_generator.class_indices)
print('n_class: ', n_class)






base_model = resnet.ResNet50(weights=pretrained,include_top=False)
my_resnet = tf.keras.Sequential([ base_model,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Dense(n_class)])


def test():
    class_names = list(valid_generator.class_indices.keys())
    for i in range(3):
        inputs, classes = next(iter(valid_generator))
        inputs = np.concatenate(inputs, axis=1)  # (N,224,224,3)  ==> (150,750,3)
        plt.figure(figsize=(15,35))
        plt.imshow(inputs)
        plt.title([class_names[int(x)] for x in classes])
        plt.show()


def train():
    n_epoch = 10
    initial_learning_rate = 0.01
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
                                monitor='val_accuracy',  # val loss를 기준으로...
                                factor=0.6,          # callback 호출시 lr = factor*lr
                                patience=10,         # patience epoch동안 monitor값이 좋아지지 않으면 callback 작동
                                cooldown=0,         # lr이 변경된 후, cooldonw동안에는 callback이 작동 안 한다.
                                min_lr=0.0001, 
                                verbose = 1,min_delta=0.001)
    #optimizer = tf.keras.optimizers.Adam(lr=initial_learning_rate)
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate,momentum=0.9)
    #optimizer = tf.keras.optimizers.RMSprop(lr=initial_learning_rate)
    
    my_resnet.compile(optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    model_filename = "Epoch-{epoch:02d}-{val_accuracy:.4f}"
    checkpoint_path = os.path.join('models_tf2/', model_filename)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,mode='auto',save_best_only=True,monitor='val_accuracy')    
    
    history = my_resnet.fit(train_generator,epochs=n_epoch,validation_data=valid_generator,validation_freq=1, verbose=1,callbacks=[cp_callback,lr_callback],initial_epoch=0)
    
    
    plot_history(history)

def evaluate():
    my_resnet.load_weights(tf.train.latest_checkpoint('models_tf2'))
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9)
    my_resnet.compile(optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    result = my_resnet.evaluate(valid_generator)
    
    print("loss-acc: ", result)
    
    

if __name__ == "__main__":
    #test()
    train()
    #evaluate()



    print('Done')