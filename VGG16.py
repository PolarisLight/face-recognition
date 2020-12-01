import tensorflow as tf


class VGG16(tf.keras.Model):
    """
    VGG网络由著名的牛津大学视觉组（Visual Geometry Group）2014年提出，
    并取得了ILSVRC 2014比赛分类任务的第2名（GoogleNet第一名）和定位任务的第1名。
    VGGNet整个网络都使用了同样大小的卷积核尺寸（3x3）和池化尺寸（2x2）。
    VGG网络的主要创新是采用了小尺寸的卷积核。所有卷积层都使用3x3卷积核，并且卷积的步长为1。
    为了保证卷积后的图像大小不变，对图像进行了填充，四周各填充1个像素。
    所有池化层都采用maxPool, 2x2的核，步长为2。
    全连接层有3层，分别包括4096，4096，1000个节点。
    除了最后一个全连接层之外，所有层都采用了ReLU激活函数。
    """

    def __init__(self, output_size, input_shape=(None, 224, 224, 3)):
        super().__init__()
        self.output_size = output_size
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=[3, 3],
                                            strides=1,
                                            activation="relu",
                                            padding='same',
                                            input_shape=input_shape,
                                            use_bias=True)
        self.conv2 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=[3, 3],
                                            strides=1,
                                            activation="relu",
                                            padding='same',
                                            use_bias=True)
        self.maxPooling1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv3 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=[3, 3],
                                            strides=1,
                                            activation="relu",
                                            padding='same',
                                            use_bias=True)
        self.conv4 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=[3, 3],
                                            strides=1,
                                            activation="relu",
                                            padding='same',
                                            use_bias=True)
        self.maxPooling2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv5 = tf.keras.layers.Conv2D(filters=256,
                                            kernel_size=[3, 3],
                                            strides=1,
                                            activation="relu",
                                            padding='same',
                                            use_bias=True)
        self.conv6 = tf.keras.layers.Conv2D(filters=256,
                                            kernel_size=[3, 3],
                                            strides=1,
                                            activation="relu",
                                            padding='same',
                                            use_bias=True)
        self.conv7 = tf.keras.layers.Conv2D(filters=256,
                                            kernel_size=[3, 3],
                                            strides=1,
                                            activation="relu",
                                            padding='same',
                                            use_bias=True)
        self.maxPooling3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv8 = tf.keras.layers.Conv2D(filters=512,
                                            kernel_size=[3, 3],
                                            strides=1,
                                            activation="relu",
                                            padding='same',
                                            use_bias=True)
        self.conv9 = tf.keras.layers.Conv2D(filters=512,
                                            kernel_size=[3, 3],
                                            strides=1,
                                            activation="relu",
                                            padding='same',
                                            use_bias=True)
        self.conv10 = tf.keras.layers.Conv2D(filters=512,
                                             kernel_size=[3, 3],
                                             strides=1,
                                             activation="relu",
                                             padding='same',
                                             use_bias=True)
        self.maxPooling4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv11 = tf.keras.layers.Conv2D(filters=512,
                                             kernel_size=[3, 3],
                                             strides=1,
                                             activation="relu",
                                             padding='same',
                                             use_bias=True)
        self.conv12 = tf.keras.layers.Conv2D(filters=512,
                                             kernel_size=[3, 3],
                                             strides=1,
                                             activation="relu",
                                             padding='same',
                                             use_bias=True)
        self.conv13 = tf.keras.layers.Conv2D(filters=512,
                                             kernel_size=[3, 3],
                                             strides=1,
                                             activation="relu",
                                             padding='same',
                                             use_bias=True)
        self.maxPooling5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=4096, activation="relu")
        self.dense2 = tf.keras.layers.Dense(units=4096, activation="relu")
        self.dense3 = tf.keras.layers.Dense(self.output_size, activation='softmax')

    ##tf.servering部署需要input_signature
    # @tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name='inputs')])
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxPooling1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxPooling2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxPooling3(x)

        x = self.conv8(x)

        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxPooling4(x)
        """
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.maxPooling5(x)
        """
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x
