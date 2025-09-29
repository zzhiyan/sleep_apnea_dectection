# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num,
                                            kernel_size=3,
                                            activation='swish',
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_num,
                                            kernel_size=3,
                                            activation='swish',
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv1D(filters=filter_num,
                                                       kernel_size=1,
                                                       activation='swish',
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = layers.Dropout(0.4)(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = layers.Dropout(0.4)(x)
        output = tf.keras.layers.add([residual, x])
        return output


def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))
    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))
    return res_block


def hybrid_resnet_model( input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filters=64, kernel_size=3,  strides=1,activation='swish',)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=3,)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, strides=1,activation='swish',)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=3, )(x)
    x = layers.Dropout(0.4)(x)
    x = make_basic_block_layer(filter_num=128, blocks=1)(x)
    x = make_basic_block_layer(filter_num=128, blocks=1, stride=2)(x)
    x = make_basic_block_layer(filter_num=256, blocks=1, stride=2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)
    x = layers.Dropout(0.4)(x)
    features = layers.Dense(64, activation="swish")(x)
    outputs_A = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="projection")(features)
    outputs_B = layers.Dense(2, activation="softmax", name="classifier",
                             kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    model = tf.keras.Model(
        inputs=inputs, outputs=[outputs_A, outputs_B], name="hybrid_model" )
    return model


# baseline
class resnet(tf.keras.Model):
    def __init__(self):
        super(resnet, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3,  strides=1, activation='swish', padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=3,)

        self.conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='swish',  padding="same")
        self.bn2  = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool1D(pool_size=3,)

        self.layer1 = make_basic_block_layer(filter_num=128, blocks=1)
        self.layer2 = make_basic_block_layer(filter_num=128,  blocks=1,  stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,  blocks=1,  stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
        self.fc2 = tf.keras.layers.Dense(units=2,
                                        activation=tf.keras.activations.softmax,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs, training=None, mask=None):


        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = layers.Dropout(0.4)(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = layers.Dropout(0.4)(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.avgpool(x)
        x = self.fc1(x)
        x = layers.Dropout(0.4)(x)
        output = self.fc2(x)
        return output


