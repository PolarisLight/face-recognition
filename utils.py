import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


# opencv往图片中写入中文,返回图片
def DrawChinese(img, text, positive, fontSize=20, fontColor=(0, 255, 0)):
    """
    img:numpy.ndarray,
    text:中文文本,
    positive:位置,
    fontSize:字体大小默认20,
    fontColor:字体颜色默认绿色)
    """
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("simhei.ttf", fontSize, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text(positive, text, fontColor, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体格式
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # PIL图片转cv2 图片

    return cv2charimg


class FaceLoader(object):
    def __init__(self, datadir="face\\"):
        self.datadir =datadir
        self.namelist = os.listdir(datadir)
        self.count = len(self.namelist)

    def getData(self):
        src = np.random.randint(0, self.count)
        opp = np.random.randint(0, self.count)
        while opp == src:
            opp = np.random.randint(0, self.count)
        srcdir = self.datadir + self.namelist[src]
        srcdir = glob.glob(srcdir + "\\*")
        oppdir = self.datadir + self.namelist[opp]
        oppdir = glob.glob(oppdir + "\\*")
        i = 0
        srcimg = []
        for srcname in srcdir:
            if np.random.rand() > 0.7:
                img = cv2.imread(srcname)
                srcimg.append(img)
                i += 1
                if i == 2:
                    break
            if srcname == srcdir[-2] and not i:
                img = cv2.imread(srcname)
                srcimg.append(img)
                i += 1
            if srcname == srcdir[-1] and i == 1:
                img = cv2.imread(srcname)
                srcimg.append(img)

        if np.random.rand() > 0.5:
            srcimg.reverse()
        i = 0
        for oppname in oppdir:
            i += 1
            if np.random.rand() > 0.9:
                img = cv2.imread(oppname)
                oppimg = img
                break
            if oppname == oppdir[-1]:
                img = cv2.imread(oppname)
                oppimg = img

        src = srcimg[0] / 255.0
        src = tf.convert_to_tensor(src)
        src = tf.expand_dims(src, 0, name=None)

        sameperson = srcimg[1] / 255.0
        sameperson = tf.convert_to_tensor(sameperson)
        sameperson = tf.expand_dims(sameperson, 0, name=None)

        diffperson = oppimg / 255.0
        diffperson = tf.convert_to_tensor(diffperson)
        diffperson = tf.expand_dims(diffperson, 0, name=None)

        return src, sameperson, diffperson


def get_cos_distance(X1, X2):
    # calculate cos distance between two sets
    # more similar more big
    # 求模
    X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1)))
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2)))
    # 内积
    X1_X2 = tf.reduce_sum(X1 * X2)
    X1_X2_norm = X1_norm * X2_norm
    # 计算余弦距离
    cos = X1_X2 / X1_X2_norm
    return cos


def cosine(q, a):
    pooled_len_1 = tf.sqrt(q * q)
    pooled_len_2 = tf.sqrt(a * a)
    pooled_mul_12 = q * a
    score = tf.math.divide(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
    return score


def celoss_ones(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def get_distance(X1, X2):
    return tf.abs(tf.nn.tanh(tf.square(X1) - tf.square(X2)))
