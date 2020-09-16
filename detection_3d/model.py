#!/usr/bin/env python
__copyright__ = """
Copyright (c) 2020 Tananaev Denis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""


import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Layer,
    UpSampling2D,
    BatchNormalization,
    LeakyReLU,
    MaxPool2D,
)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2


class InceptionBlockV1(Layer):
    """
    The inception v1 block
    See: https://arxiv.org/abs/1409.4842
    Arguments:
        reductions: list with two values for the 3x3 reduction and 5x5 reduction
        out_filters: list with four values for output channels for 1x1, 3x3, 5x5,
                     1x1 pool
        weight_decay: l2 weight decay
        data_format:  channels_last = [N,H,W,C], not support channel_first = [N,C,H,W]
    """

    def __init__(
        self, reductions, out_filters, weight_decay, data_format="channels_last"
    ):

        super(InceptionBlockV1, self).__init__()
        self.conv_1x1 = Conv2D(
            out_filters[0],
            (1, 1),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )
        self.conv_3x3 = Conv2D(
            out_filters[1],
            (3, 3),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )
        self.conv_5x5 = Conv2D(
            out_filters[2],
            (5, 5),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )
        self.conv_1x1_pool = Conv2D(
            out_filters[3],
            (1, 1),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )
        self.reduction_3x3 = Conv2D(
            reductions[0],
            (1, 1),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )
        self.reduction_5x5 = Conv2D(
            reductions[1],
            (1, 1),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )
        self.pool = MaxPool2D(
            (3, 3), strides=(1, 1), padding="same", data_format=data_format
        )

    def call(self, x):
        l1 = self.conv_1x1(x)
        red_3x3 = self.reduction_3x3(x)
        red_5x5 = self.reduction_5x5(x)
        pool = self.pool(x)
        l3 = self.conv_3x3(red_3x3)
        l5 = self.conv_5x5(red_5x5)
        l1_pool = self.conv_1x1_pool(pool)
        out = tf.concat([l1, l3, l5, l1_pool], axis=-1)

        return out


class EncoderV1(Layer):
    """
    The convolutional part of inception v1 encoder
    See: https://arxiv.org/abs/1409.4842
    """

    def __init__(self, name, weight_decay, data_format="channels_last"):
        super(EncoderV1, self).__init__(name=name)
        self.conv7x7_s2 = Conv2D(
            64,
            (7, 7),
            strides=(2, 2),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )  # 960 x 512
        self.pool1 = MaxPool2D(
            (3, 3), strides=(2, 2), padding="same", data_format=data_format
        )  # 480 x 256
        self.conv1x1_s1 = Conv2D(
            64,
            (1, 1),
            strides=(1, 1),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )
        self.conv3x3_s1 = Conv2D(
            192,
            (3, 3),
            strides=(1, 1),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )
        self.pool2 = MaxPool2D(
            (3, 3), strides=(2, 2), padding="same", data_format=data_format
        )  # 240x 128
        self.inception3a = InceptionBlockV1(
            reductions=[96, 16],
            out_filters=[64, 128, 32, 32],
            weight_decay=weight_decay,
            data_format=data_format,
        )
        self.inception3b = InceptionBlockV1(
            reductions=[128, 32],
            out_filters=[128, 192, 96, 64],
            weight_decay=weight_decay,
            data_format=data_format,
        )
        self.pool3 = MaxPool2D(
            (3, 3), strides=(2, 2), padding="same", data_format=data_format
        )  # 120 x 64
        self.inception4a = InceptionBlockV1(
            reductions=[96, 16],
            out_filters=[192, 208, 48, 64],
            weight_decay=weight_decay,
            data_format=data_format,
        )
        self.inception4b = InceptionBlockV1(
            reductions=[112, 24],
            out_filters=[160, 224, 64, 64],
            weight_decay=weight_decay,
            data_format=data_format,
        )
        self.inception4c = InceptionBlockV1(
            reductions=[128, 24],
            out_filters=[128, 256, 64, 64],
            weight_decay=weight_decay,
            data_format=data_format,
        )
        self.inception4d = InceptionBlockV1(
            reductions=[144, 32],
            out_filters=[112, 288, 64, 64],
            weight_decay=weight_decay,
            data_format=data_format,
        )
        self.inception4e = InceptionBlockV1(
            reductions=[160, 32],
            out_filters=[256, 320, 128, 128],
            weight_decay=weight_decay,
            data_format=data_format,
        )
        self.pool4 = MaxPool2D(
            (3, 3), strides=(2, 2), padding="same", data_format=data_format
        )  # 60 x 32
        self.inception5a = InceptionBlockV1(
            reductions=[160, 32],
            out_filters=[256, 320, 128, 128],
            weight_decay=weight_decay,
            data_format=data_format,
        )
        self.inception5b = InceptionBlockV1(
            reductions=[192, 48],
            out_filters=[384, 384, 128, 128],
            weight_decay=weight_decay,
            data_format=data_format,
        )

    def call(self, input):
        net = self.conv7x7_s2(input)
        net = self.pool1(net)
        net = self.conv1x1_s1(net)
        net_3 = self.conv3x3_s1(net)
        net = self.pool2(net_3)
        net = self.inception3a(net)
        net_3b = self.inception3b(net)
        net = self.pool3(net_3b)
        net = self.inception4a(net)
        net = self.inception4b(net)
        net = self.inception4c(net)
        net = self.inception4d(net)
        net_4e = self.inception4e(net)
        net = self.pool4(net_4e)
        net = self.inception5a(net)
        net_5b = self.inception5b(net)
        return net_5b, net_4e, net_3b, net_3


class DecoderV1(Layer):
    """
    Arbitrary decoder to fit inception v1 encoder.
    """

    def __init__(self, name, weight_decay, data_format="channels_last"):
        super(DecoderV1, self).__init__(name=name)
        self.up1 = UpSampling2D(size=(2, 2), data_format=data_format)
        self.conv1 = Conv2D(
            256,
            (3, 3),
            strides=(1, 1),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )
        self.up2 = UpSampling2D(size=(2, 2), data_format=data_format)
        self.conv2 = Conv2D(
            128,
            (3, 3),
            strides=(1, 1),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )
        self.up3 = UpSampling2D(size=(2, 2), data_format=data_format)
        self.conv3 = Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )
        self.up4 = UpSampling2D(size=(2, 2), data_format=data_format)
        self.conv4 = Conv2D(
            32,
            (1, 1),
            strides=(1, 1),
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            padding="same",
            data_format=data_format,
        )

    def call(self, input):
        net_5b, net_4e, net_3b, net_3 = input
        net = self.up1(net_5b)
        net = tf.concat([net, net_4e], axis=-1)
        net = self.conv1(net)
        net = self.up2(net)
        net = tf.concat([net, net_3b], axis=-1)
        net = self.conv2(net)
        net = self.up3(net)
        net = tf.concat([net, net_3], axis=-1)
        net = self.conv3(net)
        net = self.up4(net)
        net = self.conv4(net)
        return net


class InceptionDepth(Model):
    def __init__(self, weight_decay, data_format="channels_last"):
        super(InceptionDepth, self).__init__(name="InceptionDepth")
        self.encoder = EncoderV1(
            name="EncoderV1", weight_decay=weight_decay, data_format=data_format
        )
        self.decoder = DecoderV1(
            name="DecoderV1", weight_decay=weight_decay, data_format=data_format
        )
        self.final_layer = Conv2D(
            1,
            (1, 1),
            strides=(1, 1),
            activation=None,
            kernel_regularizer=None,
            padding="same",
            data_format=data_format,
        )

    def call(self, x, training):
        x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)
        x = self.final_layer(x)
        return x
