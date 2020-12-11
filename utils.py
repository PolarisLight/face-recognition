import csv
import glob
import os

import cv2
import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image, ImageDraw, ImageFont


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


class FaceDetector(object):
    def __init__(self,
                 modelFile="models\opencv_face_detector_uint8.pb",
                 configFile="models\opencv_face_detector.pbtxt",
                 dir=None):
        if dir:
            modelFile = dir + "_uint8.pb"
            configFile = dir + ".pbtxt"
        self.net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    def detectFace(self, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                     [104, 117, 123], False, False)

        self.net.setInput(blob)
        detections = self.net.forward()
        bboxes = []
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                """
                cv2.rectangle(frameOpencvDnn,
                              (x1, y1),
                              (x2, y2),
                              (0, 255, 0),
                              int(round(frameHeight / 150)),
                              8)
                """
                faces.append(frameOpencvDnn[x1:x2, y1:y2])
        return frameOpencvDnn, bboxes, faces


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
    def __init__(self, traindatadir="face\\", testdatadir="face\\"):
        self.traindatadir = traindatadir
        self.trainnamelist = os.listdir(traindatadir)
        self.testdatadir = testdatadir
        self.testnamelist = os.listdir(testdatadir)
        if self.traindatadir == testdatadir:
            self.traincount = int(len(self.trainnamelist) * 0.7)
            self.testcount = len(self.trainnamelist) - self.traincount
        else:
            self.traincount = len(self.trainnamelist)
            self.testcount = len(self.testnamelist)

    def getTrainData(self, showpic=False, input_shape=(224, 224, 3)):
        src = np.random.randint(0, self.traincount)
        opp = np.random.randint(0, self.traincount)
        while opp == src:
            opp = np.random.randint(0, self.traincount)

        srcdir = self.traindatadir + self.trainnamelist[src]
        srcdir = glob.glob(srcdir + "\\*")
        oppdir = self.traindatadir + self.trainnamelist[opp]
        oppdir = glob.glob(oppdir + "\\*")

        srcimg = []

        srcnum1 = np.random.randint(0, len(srcdir))
        srcnum2 = np.random.randint(0, len(srcdir))
        while srcnum1 == srcnum2:
            srcnum2 = np.random.randint(0, len(srcdir))

        srcimg.append(cv2.imread(srcdir[srcnum1]))
        srcimg.append(cv2.imread(srcdir[srcnum2]))

        if np.random.rand() > 0.5:
            srcimg.reverse()

        opp = np.random.randint(0, len(oppdir))
        oppimg = cv2.imread(oppdir[opp])

        if showpic:
            cv2.imshow("0", srcimg[0])
            cv2.imshow("1", srcimg[1])
            cv2.imshow("2", oppimg)
            cv2.waitKey(1000)
        # input resize
        srcimg[0] = cv2.resize(srcimg[0], (input_shape[0], input_shape[1]))
        srcimg[1] = cv2.resize(srcimg[1], (input_shape[0], input_shape[1]))
        oppimg = cv2.resize(oppimg, (input_shape[0], input_shape[1]))

        # normalization
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

    def getTestData(self, showpic=False):
        src = np.random.randint(0, self.testcount)
        opp = np.random.randint(0, self.testcount)
        while opp == src:
            opp = np.random.randint(0, self.testcount)

        srcdir = self.testdatadir + self.testnamelist[self.traincount + src + 1]
        srcdir = glob.glob(srcdir + "\\*")
        oppdir = self.testdatadir + self.testnamelist[self.traincount + opp + 1]
        oppdir = glob.glob(oppdir + "\\*")

        srcimg = []

        srcnum1 = np.random.randint(0, len(srcdir))
        srcnum2 = np.random.randint(0, len(srcdir))
        while srcnum1 == srcnum2:
            srcnum2 = np.random.randint(0, len(srcdir))

        srcimg.append(cv2.imread(srcdir[srcnum1]))
        srcimg.append(cv2.imread(srcdir[srcnum2]))

        if np.random.rand() > 0.5:
            srcimg.reverse()

        opp = np.random.randint(0, len(oppdir))
        oppimg = cv2.imread(oppdir[opp])

        if showpic:
            cv2.imshow("0", srcimg[0])
            cv2.imshow("1", srcimg[1])
            cv2.imshow("2", oppimg)
            cv2.waitKey(1000)

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
    if isinstance(X1, tf.Tensor):
        X1 = tf.convert_to_tensor(X1)
    if isinstance(X2, tf.Tensor):
        X2 = tf.convert_to_tensor(X2)
    X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1)))
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2)))
    # 内积
    X1_X2 = tf.reduce_sum(X1 * X2)
    X1_X2_norm = X1_norm * X2_norm
    # 计算余弦距离
    cos = X1_X2 / X1_X2_norm
    return cos


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


class FaceDatabase(object):
    def __init__(self, file_dir="face_database\\", databasename="my_face_encode", encodefunc=None):
        self.known_face_encodings = []
        self.known_face_names = []
        self.knowm_face = {}
        if encodefunc == None:
            import face_recognition
            self.encodefunction = face_recognition.face_encodings
        else:
            self.encodefunction = encodefunc
        read_file = glob.glob(file_dir + "*")
        lines = 0
        if os.path.exists(databasename + '.csv'):
            with open(databasename + '.csv') as f:
                f_csv = csv.reader(f)
                for _ in f_csv:
                    lines += 1
                f.close()
            with open(databasename + '.csv') as f:
                f_csv = csv.reader(f)
                with tqdm.tqdm(total=lines) as pbar:
                    for row in f_csv:
                        self.known_face_names.append(row[0])
                        face_encoding = []
                        for x in row[1:]:
                            face_encoding.append(float(x))
                        self.known_face_encodings.append(face_encoding)
                        self.knowm_face[row[0]] = face_encoding
                        pbar.set_description("now loading %s" % row[0])
                        pbar.update(1)
                f.close()
        with open(databasename + '.csv', 'a', newline='') as f:
            f_csv = csv.writer(f)
            with tqdm.tqdm(total=len(read_file)) as pbar:
                for imgname in read_file:
                    list = imgname.split("\\")
                    filename = list[-1]
                    name = filename.split(".")[0]
                    if name not in self.known_face_names:
                        print(imgname)
                        img = cv_imread(imgname)
                        img = cv2.resize(img, (224, 224))
                        src = img / 255.0
                        src = tf.convert_to_tensor(src)
                        img = tf.expand_dims(src, 0, name=None)
                        face_encoding = self.encodefunction(img)
                        face_encoding = tf.reshape(face_encoding, -1).numpy().tolist()
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                        self.knowm_face[name] = face_encoding
                        pbar.set_description("now writing %s" % name)
                        row = [name]
                        for x in face_encoding:
                            row.append(x)
                        f_csv.writerow(row)
                    else:
                        pbar.set_description("skipping %s" % name)
                    pbar.update(1)
            f.close()

    def getFaceDistance(self, img):
        face_distance = []
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = tf.convert_to_tensor(img)
        img = tf.expand_dims(img, 0, name=None)
        src_face_code = self.encodefunction(img)
        for face_name in self.known_face_names:
            distance = get_cos_distance(src_face_code,
                                        self.knowm_face.get(face_name))
            distance = distance / 2 + 0.5
            face_distance.append(distance)
        return face_distance
