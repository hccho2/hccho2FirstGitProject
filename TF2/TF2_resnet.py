
'''
tensorflow.tf.keras.applications  ---> resnet50부터 있고, resnet18이 없다.


1. ResNetTypeI, ResNetTypeII 로 resent18,.. resnet152   ---> https://github.com/calmisential/TensorFlow2.0_ResNet(bisa가 좀 잘몯 되어 있다. 수정해서 반영함)

2. 두번째 방식은 ResNetType1만 가능하다.  ---> 핸드온 머신러닝 14장 코드
MyResnet


3. 수정 Resnet(ResNetX): cifar10 data에 구조를 변경한 모델.   <----    https://github.com/kuangliu/pytorch-cifar
BasicBlockX, BottleneckX, ResNetX


1,2는 resnet18로 2개 비교해 보면 동일하다.

'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from functools import partial

NUM_CLASSES = 10

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride,use_bias=False))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same',use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same',use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same',use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride,use_bias=False))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))

    return res_block
class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same",use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=None)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same",use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=None)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


def resnet_18():
    return ResNetTypeI(layer_params=[2, 2, 2, 2])


def resnet_34():
    return ResNetTypeI(layer_params=[3, 4, 6, 3])


def resnet_50():
    return ResNetTypeII(layer_params=[3, 4, 6, 3])


def resnet_101():
    return ResNetTypeII(layer_params=[3, 4, 23, 3])


def resnet_152():
    return ResNetTypeII(layer_params=[3, 8, 36, 3])

###############################################################
###############################################################

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,padding="SAME", use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


class MyResnet(tf.keras.Model):
    def __init__(self, input_shape, layers):
        super().__init__()
        
        # layers: [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3  ---> resnet34
        # layers: [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
        
        self.model = tf.keras.models.Sequential()
        self.model.add(DefaultConv2D(64, kernel_size=7, strides=2, input_shape=input_shape))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Activation("relu"))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
        prev_filters = 64
        for filters in layers:
            strides = 1 if filters == prev_filters else 2
            self.model.add(ResidualUnit(filters, strides=strides))
            prev_filters = filters
        self.model.add(tf.keras.layers.GlobalAvgPool2D())
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))      
        
        self.built = True
        def call(self,x, training=None):
            return self.model(x,training)

###########################################################
###########################################################
class BasicBlockX(tf.keras.Model): # 논문의 resnet구조로 다르다.
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockX, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(planes,kernel_size=3, strides=stride, padding="SAME", use_bias=False)
        
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(planes,kernel_size=3, strides=1, padding="SAME", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.shortcut = tf.keras.models.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut.add(tf.keras.layers.Conv2D(self.expansion*planes,kernel_size=1, strides=stride, padding="SAME", use_bias=False))
            self.shortcut.add(tf.keras.layers.BatchNormalization())

    def call(self, x, training=None):
        out = tf.nn.relu(self.bn1(self.conv1(x),training))
        out = self.bn2(self.conv2(out),training)
        out += self.shortcut(x,training)
        out = tf.nn.relu(out)
        return out


class BottleneckX(tf.keras.Model):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckX, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=1, strides=1, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=stride, padding="SAME", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.conv3 = tf.keras.layers.Conv2D(self.expansion*planes, kernel_size=1,strides=1, use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.shortcut = tf.keras.models.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut.add(tf.keras.layers.Conv2D(self.expansion*planes, kernel_size=1, strides=stride, use_bias=False))
            self.shortcut.add(tf.keras.layers.BatchNormalization())

    def call(self, x, training=None):
        out = tf.nn.relu(self.bn1(self.conv1(x),training))
        out = tf.nn.relu(self.bn2(self.conv2(out),training))
        out = self.bn3(self.conv3(out),training)
        out += self.shortcut(x,training)
        out = tf.nn.relu(out)
        return out


class ResNetX(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetX, self).__init__()
        self.in_planes = 64

        self.conv1 = tf.keras.layers.Conv2D(64,kernel_size=3, strides=1, padding="SAME", use_bias=False) 
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.flatten = tf.keras.layers.Flatten()
        self.linear = tf.keras.layers.Dense(units=num_classes)
        
        self.built = True

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return tf.keras.models.Sequential(layers)
    
    # 단순히 super를 build를 call하는 것이면, 만들지 않아도 된다.
    def build(self,input_shape):
        #super(ResNetX, self).build(input_shape)
        super().build(input_shape)
    
    def call(self, x,training=None):
        out = tf.nn.relu(self.bn1(self.conv1(x),training))
        out = self.layer1(out,training)
        out = self.layer2(out,training)
        out = self.layer3(out,training)
        out = self.layer4(out,training)
        out = tf.nn.avg_pool2d(out, ksize=4,strides=4,padding='VALID')
        out = self.flatten(out)
        out = self.linear(out)
        return out


def ResNetX18():
    return ResNetX(BasicBlockX, [2, 2, 2, 2])
def ResNetX34():
    return ResNetX(BasicBlockX, [3, 4, 6, 3])
def ResNetX50():
    return ResNetX(BottleneckX, [3, 4, 6, 3])
def ResNetX101():
    return ResNetX(BottleneckX, [3, 4, 23, 3])

input_shape=[32, 32, 3]

model = MyResnet(input_shape=input_shape, layers=[64] * 2 + [128] * 2 + [256] * 2 + [512] * 2)

model.summary()


#######
model2 = resnet_18()
model2.build(input_shape=(None,32,32,3))
model2.summary()

# #######
print('==========ResNetX18=========')
model3 = ResNetX18()
model3.build(input_shape=(None,32,32,3))
model3.summary()

# #######
print('==========ResNet50==========')
model4 = ResNetX50()
model4.build(input_shape=(None,32,32,3))
model4.summary()

