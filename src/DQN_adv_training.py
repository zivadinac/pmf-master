import numpy as np
import gym
import tensorflow as tf
import DQN
from models import AttackModel
# adversarial training
env = gym.make('PongNoFrameskip-v4')
tf.reset_default_graph()
sess = tf.Session()
dqnAgent = DQN.DQNAgent(env, sess) # , "../ckpts/dqn/pong_adv_training/dqn_episode1050.ckpt")

adversary = AttackModel(sess, dqnAgent)
adversary.setupAttack("gauss", eps=0.015)
adversary.setAttackProb(0.5)
testRewards, testLengths, _, _ = dqnAgent.test(2)

#lengths, rewards, losses = dqnAgent.train(adversary=adversary, checkpointFolder = "../ckpts/dqn/pong_adv_training/", epNum=2000) #, epStart=1051, eps=MIN_EPS)

dqnAttack = AttackModel(dqnAgent)
dqnAttack.setupAttack("fgsm", eps=0.01)
dqnAttack.setAttackProb(1.0)

testRewards, testLengths, attNums, _ = dqnAgent.test(1, adversary=dqnAttack, render=True)
print("Mean reward = " + str(np.mean(testRewards)))
