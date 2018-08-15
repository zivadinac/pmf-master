import numpy as np
import tensorflow as tf
from cleverhans.model import Model
import cleverhans.attacks as atts
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from skimage.filters import median
import utils

class AttackModel(Model):

    def __init__(self, model, prob = 0.0, actionProbThr = 1.0):
        super(AttackModel, self).__init__()
        self.sess = model.sess
        self.layer_names = ['inputs', 'logits', 'probs']
        self.layers = [model.inputs, model.logits, model.probs]
        self.input_shape = tuple(model.inputs.shape.as_list())

        self.attackModels = {}
        self.attackModels["lbfgs"] = atts.LBFGS(self, "tf", self.sess)
        self.attackModels["fgsm"] = atts.FastGradientMethod(self, "tf", self.sess)
        self.attackModels["basicIt"] = atts.BasicIterativeMethod(self, "tf", self.sess)
        self.attackModels["pgd"] = atts.MadryEtAl(self, "tf", self.sess)
        self.attackModels["momenIt"] = atts.MomentumIterativeMethod(self, "tf", self.sess)
        self.attackModels["jsma"] = atts.SaliencyMapMethod(self, "tf", self.sess)
        self.attackModels["cwl2"] = atts.CarliniWagnerL2(self, "tf", self.sess)
        self.attackModels["ead"] = atts.ElasticNetMethod(self, "tf", self.sess)
        self.attackModels["deepfool"] = atts.DeepFool(self, "tf", self.sess)
        self.attackModels["spsa"] = atts.SPSA(self, "tf", self.sess)
        self.attackModels["featureAdvs"] = atts.FastFeatureAdversaries(self, "tf", self.sess)
        
        self.availableAttacks = list(self.attackModels.keys())
        self.availableAttacks.append("gauss")
        self.gaussEps = 0.0
        self.attack = None
        self.lastAttack = None
        self.attackProb = prob
        self.actionProbThr = actionProbThr

    def toAttack(self, actionProbs):
        if self.attackProb != 0.0:
            return np.random.rand() < self.attackProb
        else:
            return np.max(actionProbs) - np.min(actionProbs) >= self.actionProbThr

    def setAttackProb(self, attackProb):
        self.attackProb = attackProb
        self.actionProbThr = 1.0

    def setActionProbThr(self, actionProbThr):
        self.actionProbThr = actionProbThr
        self.attackProb = 0.0

    def get_probs(self, x, **kwargs):
        return self.layers[2]

    def get_logits(self, x, **kwargs):
        return self.layers[1]

    def fprop(self, x):
        return dict(zip(self.layer_names, self.layers))

    def get_input(self):
        return self.layers[0]

    def get_action_scores(self):
        return self.layers[1]
    
    def setupAttack(self, attack, **kwargs):
        if self.availableAttacks.count(attack) == 0:
            raise ValueError("Unknown attack type '" + str(attack) + "'.")

        if attack != self.lastAttack:
            if attack == "gauss":
                self.gaussEps = kwargs["eps"]
                self.attack = None
            else:
                self.attack = self.attackModels[attack].generate(self.layers[0], **kwargs)

            self.lastAttack = attack

    def runAttack(self, x):
        if self.lastAttack == "gauss":
            return x + self.gaussEps * np.random.randn(*x.shape)

        return self.sess.run(self.attack, feed_dict={self.get_input(): x})

    def run(self, x):
        return self.sess.run(self.layers[1], feed_dict={self.get_input(): x})

class AdvDetector:

    def __init__(self, dqn, thr):
        self.dqn = dqn
        self.thr = thr
        self.resetStats()

    def isAdv(self, x, gt):
        guess = self._score(x) > self.thr
        self._updateStats(guess, gt)
        return guess

    def getStats(self):
        """ Return precision, accuracy, recall and f-measure """
        precision = 0.0
        accuracy = 0.0
        recall = 0.0
        f = 0.0
        try:
            precision = self.tpCounter / (self.tpCounter + self.fpCounter)
            accuracy = (self.tpCounter + self.tnCounter) / (self.tpCounter + self.fpCounter + self.tnCounter + self.fnCounter)
            recall = self.tpCounter / (self.tpCounter + self.fnCounter)
            f = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError as err:
            print("ZeroDivisionError in AdvDetector.getStats(): " + str(err))

        return (precision, accuracy, recall, f)

    def printStats(self):
        p, a, r, f = self.getStats()
        print("Precision = " + str(p) + ", recall = " + str(r) + ", accuracy = " + str(a) + ", f-measure = " + str(f))

    def resetStats(self):
        self.fpCounter = 0
        self.tpCounter = 0
        self.fnCounter = 0
        self.tnCounter = 0

    def _squeeze(self, x):
        xn = x - np.min(x)
        xn = xn / np.max(x)
        return np.maximum(np.sign(xn - 0.5), 0)

    def _median(self, x):
        xm = np.zeros_like(x)
        xm[0, ..., 0] =  median(x[0, ..., 0])
        xm[0, ..., 1] =  median(x[0, ..., 1])
        xm[0, ..., 2] =  median(x[0, ..., 2])
        xm[0, ..., 3] =  median(x[0, ..., 3])
        return xm

    def _score(self, x):
        sq = self._squeeze(x)
        md = self._median(x)
        xx = np.concatenate((x, sq, md), 0)
        probs = self.dqn.runProbs(xx)
        sqScore = np.linalg.norm(probs[0] - probs[1], 1)
        mdScore = np.linalg.norm(probs[0] - probs[2], 1)
        return np.maximum(sqScore, mdScore)

    def _updateStats(self, guess, gt):
        """ 1 - adv state
            0 - legitimate state """
        if guess:
            if gt:
                self.tpCounter += 1
            else:
                self.fpCounter += 1
        else:
            if gt:
                self.fnCounter += 1
            else:
                self.tnCounter += 1

def defaultTrainPrint(ep, epochNum, loss, acc):
    print("Finished epoch " + str(ep) + " / " + str(epochNum) + " avg loss = " + str(loss) + " acc = " + str(acc))

class SubstituteModel():
    
    def __init__(self, env, sess, ckptFile = None, trainPrint = defaultTrainPrint):
        self.namescope = "substitute"
        self.sess = sess
        self.env = env
        self.trainPrint = trainPrint
        if ckptFile != None:
            self.probs, self.logits, self.inputs, self.labels, self.learningRate, self.update, self.loss = self.loadSubstitute(sess, ckptFile)
        else:
            self.probs, self.logits, self.inputs, self.labels, self.learningRate, self.update, self.loss = self.createSubstituteNet(env)

    def createSubstituteNet(self, env):
        """ Create tf graph for q function approximation. """
        with tf.variable_scope(self.namescope):
            initializer = tf.contrib.layers.xavier_initializer()
            inputs = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name="subInputs")
            labels = tf.placeholder(shape=[None, ], dtype=tf.uint8, name="subLabels")
            learningRate = tf.placeholder(shape=[], dtype=tf.float32, name="subLr")

            conv1 = tf.layers.conv2d(inputs=inputs, kernel_size=[7, 7], filters=30, strides=2, padding="valid", activation=tf.nn.relu, kernel_initializer=initializer, name="subConv1")
            conv2 = tf.layers.conv2d(inputs=conv1, kernel_size=[5, 5], filters=30, strides=2, padding="valid", activation=tf.nn.relu, kernel_initializer=initializer, name="subConv2")
            pool3 = tf.nn.pool(conv2, [2,2], "MAX", "VALID", strides=[2,2], name="subPool3")

            fc4 = tf.layers.dense(inputs=tf.reshape(pool3,[-1, 9*9*30]), units=150, kernel_initializer=initializer, activation=tf.nn.relu, use_bias=False, name="subFc4")
            fc5 = tf.layers.dense(inputs=fc4, units=env.action_space.n, kernel_initializer=initializer, activation=tf.nn.relu, use_bias=False, name="subFc5")
            logits = fc5
            probs = tf.nn.softmax(logits, name="subProbs")

            loss = tf.identity(tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, env.action_space.n), logits=logits), name="subLoss")
            opt = tf.train.AdamOptimizer(learning_rate=learningRate, name="subOpt")
            updateNet = opt.minimize(loss, name="subUpdateNet")
            print("Initializing substitute variables.")
            self.sess.run(tf.global_variables_initializer())

        return probs, logits, inputs, labels, learningRate, updateNet, loss

    def loadSubstitute(self, sess, checkpoint):
        #saver = tf.train.Saver(name="subSaver", var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.namescope))
        saver = tf.train.import_meta_graph(checkpoint + ".meta", clear_devices=True)
        saver.restore(sess, checkpoint)
        logits = sess.graph.get_tensor_by_name(self.namescope + "/subFc5/Relu:0")
        probs = sess.graph.get_tensor_by_name(self.namescope + "/subProbs:0")
        inputs = sess.graph.get_tensor_by_name(self.namescope + "/subInputs:0")
        labels = sess.graph.get_tensor_by_name(self.namescope + "/subLabels:0")
        lr = sess.graph.get_tensor_by_name(self.namescope + "/subLr:0")
        update = sess.graph.get_operation_by_name(self.namescope + "/subUpdateNet")
        loss = sess.graph.get_tensor_by_name(self.namescope + "/subLoss:0")
        return probs, logits, inputs, labels, lr, update, loss

    def train(self, X, Y, lr, epochNum, batchSize, augStart, augSize, augBatchSize, dqn, checkpointFolder, epochStart = 0):
        lrr = lr
        accuracy = tf.metrics.accuracy(self.labels, tf.argmax(self.logits, axis=1), name="accuracy")
        self.sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, "accuracy")))
        saver = tf.train.Saver(name="subSaver", var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.namescope))
        losses = []
        accs = []
        xx = X
        yy = Y

        for ep in range(epochStart, epochNum):
            if ep > 0 and ep % 20 == 0:
                lr *= 0.4

            if ep == augStart:
                lr = lrr * 0.4

            if ep >= augStart and dqn != None:
                augInds = np.random.permutation(X.shape[0])[:augSize]
                xx = X[augInds]
                yy = Y[augInds]
                xx, yy = self.augmentSubstituteData(xx, yy, dqn, augBatchSize, [0.1, 0.20])
                # xx = np.vstack([xx, X])
                # yy = np.hstack([yy, Y])

            mbInds = utils.getMinibatchInds(batchSize, np.random.permutation(xx.shape[0]))

            for mbi in mbInds:
                mbX = xx[mbi]
                mbY = yy[mbi]
                _, l = self.sess.run([self.update, self.loss], feed_dict={self.inputs:mbX, self.labels:mbY, self.learningRate:lr})
                losses.append(l)

            testInds = np.random.permutation(X.shape[0])[:2000]
            acc = self.sess.run(accuracy, feed_dict={self.inputs:X[testInds], self.labels:Y[testInds]})
            self.trainPrint(ep, epochNum, np.mean(losses), acc[1])
            accs.append(acc)
            saver.save(self.sess, checkpointFolder + "epoch_" + str(ep) + ".ckpt")

        saver.save(self.sess, checkpointFolder + "final.ckpt")
        return losses, accs

    def augmentSubstituteData(self, X, Y, dqn, batchSize, lmbdas, verbose=False):
        grads = jacobian_graph(self.logits, self.inputs, self.env.action_space.n)
        epNum = len(lmbdas)
        Xa = X
        Ya = Y

        for ep in range(epNum):
            mbInds = utils.getMinibatchInds(batchSize, np.arange(X.shape[0]))
            lmbda = lmbdas[ep]

            for i, mbi in enumerate(mbInds):
                mbX = X[mbi]
                mbY = Y[mbi]

                mbXa = jacobian_augmentation(self.sess, self.inputs, mbX, mbY, grads, lmbda)
                mbXa = mbXa[mbX.shape[0]:]
                mbYa = dqn.run(mbXa)
                mbYa = np.argmax(mbYa, axis=1)
                Xa = np.vstack([Xa, mbXa])
                Ya = np.hstack([Ya, mbYa])
                del mbXa
                del mbYa
                if verbose:
                    print("Finished minibatch " + str(i) + " / " + str(len(mbInds)) + " in epoch " + str(ep) + ". Num examples = " + str(Xa.shape[0]))

            if verbose:
                print("Finished epoch " + str(ep))

        return Xa, Ya
