import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import inception
slim = tf.contrib.slim
MODELS_PATH=""
import sys
sys.path.append(MODELS_PATH+"research/slim")
from datasets import imagenet
from skimage.io import imread, imsave
from skimage import img_as_float
from utils import oneHot
from models import AttackModel
#import matplotlib.pyplot as plt

class Classifier:

    def __init__(self, sess, ckptFile):
        self.sess = sess
        self.inputs = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
        self.classes = imagenet.create_readable_names_for_imagenet_labels()

        with self.sess.as_default():
            with tf.contrib.slim.arg_scope(inception.inception_v1_arg_scope()):
                self.logits, self.net = inception.inception_v1(self.inputs, 1001, is_training=False)
                self.probs = tf.nn.softmax(self.logits)
                initFn = slim.assign_from_checkpoint_fn(ckptFile, slim.get_model_variables("InceptionV1"))
                initFn(self.sess)

    def run(self, im, verbose=True):
        if len(im.shape) == 3:
            imm = np.expand_dims(im, 0)
        elif len(im.shape) == 4:
            imm = im
        else:
            raise ValueError("im must have shape (None, 224, 224, 3) or (224, 224, 3).")

        probs = self.sess.run(self.probs, feed_dict={self.inputs:imm})[0]
        classId = np.argmax(probs)
        classs = self.classes[classId]
        p = np.max(probs)

        if verbose:
            print(classs, "(" + str(p) + ")")

        return classs, classId, p

####################################################################################################

sess = tf.Session()
model = Classifier(sess, "../ckpts/inception_v1/inception_v1.ckpt")

examplePath = "../data/wolf.png"
example = img_as_float(imread(examplePath)).astype(np.float32)

attack = AttackModel(model)
attack.setupAttack("fgsm", eps=0.007)
attack.setAttackProb(1)

advExample = attack.runAttack(np.expand_dims(example, 0))[0]
advExample = advExample - np.min(advExample)
advExample = advExample / np.max(advExample)
imsave(examplePath+"_adv.png", advExample)

c, _, p = model.run(example)
advC, _, advP = model.run(advExample)
