"""DQN example"""
from collections import deque
from copy import copy
from time import time
import numpy as np
import tensorflow as tf
import utils
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym_wrappers import AdversarialWrapper

LR = 0.0001
GAMMA = 0.99
NUM_EPISODES = 2500
TRAINING_START = 5
EXPERIENCE_SIZE = 50000
MINIBATCH_SIZE = 32
EPS = 1.0
EPS_DECAY_INTERVAL = 10
EPS_DECAY_RATE = 0.75
MIN_EPS = 0.02
CHECKPOINT_FOLDER = "dqn_checkpoints/"
CHECKPOINT_INTERVAL = 150
PRINT_INTERVAL = 10


def getRandomMinibatch(experience, minibatchSize):
    if len(experience) == 0:
        return []

    l = min(minibatchSize, len(experience))
    statesShape = experience[0][0].shape
    startStates = np.zeros((l, statesShape[1], statesShape[2], statesShape[3]))
    actions = np.zeros((l, 1))
    rewards = np.zeros((l, 1))
    endStates = np.zeros((l, statesShape[1], statesShape[2], statesShape[3]))
    dones = np.zeros((l, 1))

    for i, ind in enumerate(np.random.choice(len(experience), l)):
        ss, a, r, es, d = experience[ind]
        startStates[i, :, :, :] = ss
        actions[i, 0] = a
        rewards[i, 0] = r
        endStates[i, :, :, :] = es
        dones[i, 0] = int(d)

    return startStates, actions, rewards, endStates, dones

def computeMinibatchTargets(actions, rewards, dones, gamma, actionScoresSS, actionScoresES):
    notDones = (dones == 0).astype(int)
    t = rewards + notDones * (gamma * np.expand_dims(np.max(actionScoresES, 1), 1))
    targets = copy(actionScoresSS)
    inds = np.arange(actions.shape[0])
    targets[inds, actions.astype(int).squeeze()] = t.squeeze()
    return targets

class DQNAgent(): 
    def __init__(self, env, sess, ckptFile = None, lr = 0.0001):
        self.sess = sess

        if type(env) != AdversarialWrapper:
            self.env = AdversarialWrapper(env)
        else:
            self.env = env

        self.frameSkip = 4
        if ckptFile != None:
            self.probs, self.logits, self.update, self.loss, self.inputs, self.target = self.loadDQN(sess, ckptFile)
        else:
            self.probs, self.logits, self.update, self.loss, self.inputs, self.target = self.createDQN(env, lr) # TODO create placeholder for lr
            self.sess.run(tf.global_variables_initializer())

    def createDQN(self, env, learningRate):
        """ Create tf graph for q function approximation. """
        initializer = tf.contrib.layers.xavier_initializer()
        inputs = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32)

        conv1 = tf.layers.conv2d(inputs=inputs, kernel_size=[8, 8], filters=16, strides=4, padding="valid", activation=tf.nn.relu, kernel_initializer=initializer, name="conv1")
        conv2 = tf.layers.conv2d(inputs=conv1, kernel_size=[4, 4], filters=32, strides=2, padding="valid", activation=tf.nn.relu, kernel_initializer=initializer, name="conv2")
        conv3 = tf.layers.conv2d(inputs=conv2, kernel_size=[3, 3], filters=32, strides=1, padding="valid", activation=tf.nn.relu, kernel_initializer=initializer, name="conv3")

        fc4 = tf.layers.dense(inputs=tf.reshape(conv3,[-1, 7*7*32]), units=256, kernel_initializer=initializer, activation=tf.nn.relu, name="fc4")
        fc5 = tf.layers.dense(inputs=fc4, units=env.action_space.n, kernel_initializer=initializer, name="fc5")
        predictA = tf.argmax(fc5, axis=1, name="predictA")

        target = tf.placeholder(shape=[None, env.action_space.n], dtype=tf.float32, name="target")
        loss = tf.losses.huber_loss(labels=fc5, predictions=target)
        opt = tf.train.AdamOptimizer(learning_rate=learningRate, name="opt")
        updateNet = opt.minimize(loss, name="updateNet")

        return tf.nn.softmax(fc5), fc5, updateNet, loss, inputs, target

    def loadDQN(self, sess, checkpoint = "./dqn_checkpoints/dqn_final.ckpt"):
        saver = tf.train.import_meta_graph(checkpoint + ".meta")
        saver.restore(sess, checkpoint)
        logits = sess.graph.get_tensor_by_name("fc5/BiasAdd:0")
        probs = tf.nn.softmax(logits)
        inputs = sess.graph.get_tensor_by_name("Placeholder:0")
        update = sess.graph.get_operation_by_name("updateNet")
        loss = sess.graph.get_tensor_by_name("huber_loss/value:0")
        target = sess.graph.get_tensor_by_name("target:0")
        return probs, logits, update, loss, inputs, target

    def _attack(self, adversary, frames, actionProbs):
        """ Create adversarial state given adversary.
            Return wheter adversary state was created. """
        if adversary != None and adversary.toAttack(actionProbs):
            inputFrames = adversary.runAttack(frames)
            advDiff = inputFrames - frames
            advDiff = utils.createAdvDiffFrame(advDiff[0,:,:,3])
            self.env.setAdvDiff(advDiff)
            return True, inputFrames
        
        self.env.setAdvDiff(None)
        return False, frames

    def goalReached(self, rewards):
        """ Check if game is solved."""
        return len(rewards) >= 100 and np.mean(rewards[-100:]) >= 18

    def train(self, gamma = GAMMA, learningRate = LR, eps = EPS, epsDecayInterval = EPS_DECAY_INTERVAL, epsDecayRate = EPS_DECAY_RATE, minEps = MIN_EPS, epNum = NUM_EPISODES, epStart = 0, trainingStart = TRAINING_START, experienceSize = EXPERIENCE_SIZE, minibatchSize = MINIBATCH_SIZE, adversary = None, checkpointFolder = CHECKPOINT_FOLDER, checkpointInterval = CHECKPOINT_INTERVAL, printInterval = PRINT_INTERVAL):
        """ Train agent to interact with given environment env using approximation of Q function.
        Returns action approximation function. """

        allActions = np.asarray(range(self.env.action_space.n))
        saver = tf.train.Saver()
        experience = deque([], experienceSize)

        episodeLengths = []
        episodeLengthsSeconds = []
        episodeRewards = []
        attacksNumbers = []
        losses = []

        trainingStart = epStart + trainingStart
        for i in range(epStart, epNum):
            s = utils.preprocess(self.env.reset())
            frames = np.expand_dims(np.repeat(s, 4, 2), 0)
            done = False
            episodeLength = 0
            episodeReward = 0.0
            attNum = 0

            episodeStartTime = time()
            while not done:
                actionScores, actionProbs = self.sess.run([self.logits, self.probs], feed_dict={self.inputs:frames})
                a = np.random.choice(allActions, p=utils.epsGreedyProbs(actionScores[0], eps))
                self._attack(adversary, frames, actionProbs)

                for j in range(self.frameSkip):
                    sj, r, done, _ = self.env.step(a)
                    sj = utils.preprocess(sj)
                    episodeLength += 1
                    episodeReward += r

                framesJ = utils.pushframe(frames, sj)
                experience.append((frames, a, r, framesJ, done))
                frames = framesJ

                if i > trainingStart:
                #    actionScoresJ = sess.run(outQ, feed_dict={self.inputs:framesJ})
                    startStates, actions, rewards, endStates, dones = getRandomMinibatch(experience, minibatchSize)

                    actionScoresSS = self.sess.run(self.logits, feed_dict={self.inputs:startStates})
                    actionScoresES = self.sess.run(self.logits, feed_dict={self.inputs:endStates})
                    targets = computeMinibatchTargets(actions, rewards, dones, gamma, actionScoresSS, actionScoresES)
                    los = self.sess.run([self.loss, self.update], feed_dict={self.inputs:startStates, self.target:targets})[0]
                    losses.append(los)

            episodeEndTime = time()
            episodeLengths.append(episodeLength)
            episodeLengthsSeconds.append(episodeEndTime-episodeStartTime)
            episodeRewards.append(episodeReward)
            attacksNumbers.append(attNum)

            if eps > minEps and ((i+1) % epsDecayInterval) == 0:
                eps = eps * epsDecayRate
                print("eps decayed to " + str(eps) + " in episode " + str(i + 1) + " (" + str(sum(episodeLengths)) + "'th timestamp)")

            if (i + 1) % checkpointInterval == 0:
                saver.save(self.sess, checkpointFolder + "dqn_episode" + str(i + 1) + ".ckpt")
                print("Saved checkpoint in episode " + str(i + 1) + " with reward = " + str(episodeRewards[-1]))

            if (i + 1) % printInterval == 0:
                print(str(i + 1) + " / " + str(epNum) + " length = " + str(np.mean(episodeLengths[-10:])) + " (" + str(np.mean(episodeLengthsSeconds[-10:])) + "s) reward = " + str(np.mean(episodeRewards[-10:])) + " loss = " + str(losses[-1]))
            if self.goalReached(episodeRewards):
                print("Finished training after " + str(i + 1) + " episodes. Goal achieved.")
                break

        saver.save(self.sess, checkpointFolder + "dqn_final.ckpt")
        print("Finished training. Saved final checkpoint.")
        return episodeLengths, episodeRewards, attacksNumbers, losses

    def test(self, gamesNum = 100, adversary = None, advDetector = None, render = False, verbose = True, videoPath = None):
        """ Test trained DQN agent. """
        recordVideo = videoPath is not None
        if recordVideo:
            recorder = VideoRecorder(self.env, videoPath)

        gameRewards = []
        gameLengths = []
        attacksNumbers = []
        for i in range(gamesNum):
            done = False
            s = utils.preprocess(self.env.reset())
            frames = np.expand_dims(np.repeat(s, 4, 2), 0)
            gameReward = 0.0
            gameLength = 0
            attNum = 0
            while not done:
                actionScores, actionProbs = self.sess.run([self.logits, self.probs], feed_dict={self.inputs:frames})
                isAdvState, advFrames = self._attack(adversary, frames, actionProbs)
                if advDetector is not None:
                    advDetector.isAdv(advFrames, isAdvState)
                attNum += isAdvState

                for j in range(self.frameSkip):
                    sj, r, done, _ = self.env.step(np.argmax(actionScores))
                    gameReward += r
                    gameLength += 1
                    if render:
                        self.env.render()
                    if recordVideo:
                        recorder.capture_frame()

                frames = utils.pushframe(frames, utils.preprocess(sj))

            gameRewards.append(gameReward)
            gameLengths.append(gameLength)
            attacksNumbers.append(attNum)
            if verbose:
                print("Finished test game " + str(i+1) + " / " + str(gamesNum) + " reward = " + str(gameReward))
                print('{"metric": "loss", "value":' + str(gameReward) + '}')

        print("Agent achieved average reward of " + str(np.mean(gameRewards)) + " in " + str(gamesNum) + " games.")
        print('{"metric": "loss", "value":' + str(np.mean(gameRewards)) + '}')
        if recordVideo:
            recorder.close()

        return gameRewards, gameLengths, attacksNumbers, advDetector

    def run(self, x):
        return self.sess.run(self.logits, feed_dict={self.inputs:x})

    def runProbs(self, x):
        return self.sess.run(self.probs, feed_dict={self.inputs:x})

#env = gym.make('PongNoFrameskip-v4')
#sess = tf.InteractiveSession()
#dqnAgent = DQNAgent(env, sess, "./dqn_checkpoints/adv_training/dqn_episode150.ckpt")
#dqnAttack = AttackModel(sess, dqnAgent.inputs, dqnAgent.logits, dqnAgent.probs)
#dqnAttack.setup_fgsm(0.004)
#lengths, rewards, attacksNumbers, losses = dqnAgent.train(adversary=dqnAttack, advProb=0.3, eps=0.01, checkpointFolder = "./dqn_checkpoints/adv_training/")
#testRewards, testLengths, attacksNumbers, _ = dqnAgent.test(100, attackModel=dqnAttack, attackProb=0.3)

