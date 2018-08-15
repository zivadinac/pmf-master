import pickle
import numpy as np
import tensorflow as tf
import gym
import utils
import models
from DQN import DQNAgent

def generateSubstituteExamples(env, dqn, eps = 0.02, numPerAction = -1, gamesNum = 5):
    rewards = []
    actionNum = env.action_space.n
    allActions = np.arange(actionNum)
    xx = []
    yy = []
    it = 0
    actionCounts = np.zeros(actionNum, dtype=np.int32)

    for gn in range(gamesNum):
        done = False
        s = utils.preprocess(env.reset())
        frames = np.expand_dims(np.repeat(s, 4, 2), 0)
        gr = 0

        while not done:

            scores = dqn.run(frames)
            a = np.argmax(scores)
            xx.append(frames[0, :, :, :])
            yy.append(a)
            actionCounts[a] += 1
            it += 1

            a = np.random.choice(allActions, p=utils.epsGreedyProbs(scores[0], eps))
            for j in range(dqn.frameSkip):
                sj, r, done, _ = env.step(a)
                gr += r
            frames = utils.pushframe(frames, utils.preprocess(sj))
        rewards.append(gr)
        print("Finished game " + str(gn) + " with reward " + str(gr))

    tr = int(np.sum(np.minimum(actionCounts, numPerAction)))
    X = np.zeros((tr, 84, 84, 4))
    Y = np.zeros((tr, ), dtype=np.uint8)
    xx = np.array(xx)
    yy = np.array(yy)
    si = 0
    for a in allActions:
        xxa = xx[np.where(yy == a)[0], :, :, :]
        yya = yy[np.where(yy == a)[0]]
        inds = np.random.permutation(xxa.shape[0])[0:numPerAction]

        ei = si + inds.shape[0]
        X[si:ei, :, :, :] = xxa[inds, :, :, :]
        Y[si:ei] = yya[inds]
        si = ei
        
    return np.array(X), np.array(Y), rewards

def saveSubstituteData(X, Y, folder):
    utils.save(X, folder + "X_substitute.pck")
    utils.save(Y, folder + "Y_substitute.pck")

def loadSubstituteData(folder):
    X = utils.load(folder + "X_substitute.pck")
    Y = utils.load(folder + "Y_substitute.pck")
    return X, Y

tf.reset_default_graph()
sess = tf.Session()

env = gym.make('PongNoFrameskip-v4')
CHECKPOINT_FOLDER = "../ckpts/substitute/sub_8_18/"
SUB_DATA_FOLDER = "../data/sub_8_18/"
SUB_AUG_DATA_FOLDER = "../data/sub_8_18/aug/"
BS = 1
LR = 0.0001
EP_NUM = 50
EP_NUM_AUG = 50

#dqn = DQNAgent(env, sess, "../ckpts/dqn/pong_final/dqn_final.ckpt")
#X, Y, rewards = generateSubstituteExamples(env, dqn, numPerAction=300, gamesNum=5)
#saveSubstituteData(X, Y, SUB_DATA_FOLDER)
#X, Y = loadSubstituteData(SUB_DATA_FOLDER)

#sub = models.SubstituteModel(env, sess) #, "../ckpts/substitute/final.ckpt")
#sub.train(X, Y, LR, EP_NUM, BS, EP_NUM+1, 0, 0, None, CHECKPOINT_FOLDER)

#Xa, Ya = sub.augmentSubstituteData(X, Y, dqn, 500, [0.1, 0.2])
#saveSubstituteData(Xa, Ya, SUB_AUG_DATA_FOLDER)
sub = models.SubstituteModel(env, sess)
Xa, Ya = loadSubstituteData(SUB_AUG_DATA_FOLDER)
sub.train(Xa, Ya, LR, EP_NUM_AUG, BS, EP_NUM_AUG+1, 0, 0, None, CHECKPOINT_FOLDER)

tf.reset_default_graph()
sess.close()
sess = tf.Session()

sub = models.SubstituteModel(env, sess, "../ckpts/substitute/final.ckpt")
attack = models.AttackModel(sub)
attack.setupAttack("fgsm", eps=0.5)
attack.setAttackProb(1.0)
#attack.setActionProbThr(0.001)

dqn = DQNAgent(env, sess, "../ckpts/dqn/pong_final/dqn_final.ckpt")
rewards, lengths, attNums, _ = dqn.test(5, attack) #, render=True)
