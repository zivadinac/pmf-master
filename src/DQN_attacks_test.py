import argparse
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tensorflow as tf
import gym
import utils
import time
from models import AttackModel, SubstituteModel, AdvDetector
import cleverhans.attacks as atts
from DQN import DQNAgent

def plotAttackStats(attackStats, probs):
    attackName = attackStats["attack"]
    keys = probs.copy()
    keys.reverse()
    keys.append('timed')
    keysSrb = list(map(lambda p: 'p = ' + str(p), probs))
    keysSrb.reverse()
    keysSrb.append('страт.')
    meanRews = []
    meanAttPercs = []
    # adv stats
    precs = []
    accs = []
    recalls = []
    fms = []

    for k in keys:
        meanRews.append(attackStats[k]['meanReward'])
        meanAttPercs.append(np.round(attackStats[k]['meanAttNum'] / attackStats[k]['meanLength'] * 100, 2))
        precs.append(attackStats[k]['advDetStats'][0])
        accs.append(attackStats[k]['advDetStats'][1])
        recalls.append(attackStats[k]['advDetStats'][2])
        fms.append(attackStats[k]['advDetStats'][3])

    x = range(len(keys))
    rewsFig, rewsPlt = plt.subplots()
    rewsPlt.set_title("Учинак '" + str(attackName) + "' напада")
    rewsPlt.text(x=3.72, y=meanRews[-1] / 1.8, s=str(meanAttPercs[-1]) + '%', color="white")
    rewsPlt.set_xticks(x)
    rewsPlt.set_xticklabels(keysSrb)
    rewsPlt.bar(x, meanRews)

    detFig, detPlt = plt.subplots()
    detPlt.set_title("Откривање '" + str(attackName) + "' напада")
    detPlt.set_xticks(x)
    detPlt.set_xticklabels(keysSrb)
    detPlt.plot(x, precs, ls = 'dashed', lw = 3)
    detPlt.plot(x, accs, ls = 'dotted')
    detPlt.plot(x, recalls, ls = 'dashdot')
    detPlt.plot(x, fms, ls = 'solid')
    detPlt.legend(('Прецизност', 'Тачност', 'Одзив', 'Ф-мера'), loc='lower right')

    return rewsFig, detFig


def savePlotAttackStats(attackStats, probs, saveFolder):
        rewsFig, detFig = plotAttackStats(attackStats, probs)
        attackName = attackStats["attack"]
        rewsFig.savefig(saveFolder + attackName + "_rews.png")
        plt.close(rewsFig)
        detFig.savefig(saveFolder + attackName + "_det.png")
        plt.close(detFig)

def savePlotStats(stats, probs, saveFolder):
    for attack, attackStats in stats.items():
        savePlotAttackStats(attackStats, probs, saveFolder)

def testAdversary(gamesNum, dqnAgent, adversary, detThr):
    attDet = AdvDetector(dqnAgent, detThr)
    start = time.time()
    rs, ls, ans, attDet = dqnAgent.test(gamesNum, adversary, attDet) #, render=True)
    end = time.time()
    mt = (end-start) / gamesNum
    return rs, ls, mt, ans, attDet.getStats()

def testAttacks(dqn, attackModel, attacksConfig, gamesNum, attackProbs, actionProbThr, attDetThr, saveFolder):
    stats = {}
    for attack, params in attacksConfig.items():
        print("Testing " + attack + " attack.")
        attackModel.setupAttack(attack, **params)
        sts = {"attack": attack}

        for prob in attackProbs:
            attackModel.setAttackProb(prob)
            rewards, lengths, meanTime, attNums, attDetStats = testAdversary(gamesNum, dqn, attackModel, attDetThr)
            print(attack + "_" + str(np.round(prob, 2)) + " avgTime = " + str(meanTime) + "s" + " avgReward = " + str(np.mean(rewards)))

            sts[np.round(prob, 2)] = {"prob": prob, "rewards": rewards, "meanReward": np.mean(rewards), "lengths": lengths, "meanLength": np.mean(lengths), "meanTime": meanTime, "meanAttNum": np.mean(attNums), "advDetStats": attDetStats}

        # strategically timed attack
        attackModel.setActionProbThr(actionProbThr)
        rewards, lengths, meanTime, attNums, attDetStats = testAdversary(gamesNum, dqn, attackModel, attDetThr)

        print(attack + "_timed" + " avgTime = " + str(meanTime) + "s" + " avgReward = " + str(np.mean(rewards)))

        sts["timed"] = {"apThr": attackModel.actionProbThr, "rewards": rewards, "meanReward": np.mean(rewards), "lengths": lengths, "meanLength": np.mean(lengths), "meanTime": meanTime, "meanAttNum": np.mean(attNums), "advDetStats": attDetStats}

        stats[attack] = sts
        utils.save(sts, saveFolder + attack + ".pck")
        savePlotAttackStats(sts, attackProbs, saveFolder)
    return stats

attacks = {}
attacks["fgsm"] = {"eps": 0.01}
#attacks["jsma"] = {"eps": 0.01}
#attacks["basicIt"] = {"y":None, "eps": 0.005, "eps_iter": 0.005, "nb_iter": 3, "y_target": tf.one_hot([0], 6)}
#attacks["pgd"] = {"eps": 0.002, "eps_iter": 0.002, "nb_iter": 3}
#attacks["momenIt"] = {"eps": 0.002, "eps_iter": 0.002, "nb_iter": 3}
#attacks["cwl2"] = {'binary_search_steps': 1,
#                 'y_target': utils.oneHot(6, 0), #tf.one_hot([0], 6),
#                 'max_iterations': 100,
#                 'learning_rate': 0.1,
#                 'batch_size': 1,
#                 'initial_const': 10}
#attacks["ead"] = {"beta": 1e-4, "abort_early": True}
#attacks["deepfool"] = {"nb_candidate": 3, "nb_classes": 6}
#attacks["spsa"] = {"epsilon": 0.01, "learning_rate": 0.01, "num_steps": 100}
#attacks["featureAdvs"] = {"eps": 0.01, "eps_iter": 0.01, "layer": 'conv1', "g": 'conv1'}

gamesNum = 100
attackProbs = [1.0, 0.75, 0.5, 0.25]
actionProbThr = 0.001
attDetThr = 0.8

tf.reset_default_graph()
sess = tf.Session()
env = gym.make('PongNoFrameskip-v4')

statsFolder = "../experiments/pong/"
dqnModelPath = "../ckpts/dqn/pong_final/dqn_final.ckpt"
dqn = DQNAgent(env, sess, dqnModelPath)
attackModel = AttackModel(dqn)
stats = testAttacks(dqn, attackModel, attacks, gamesNum, attackProbs, actionProbThr, attDetThr, statsFolder)

#adv training test
attDetThr = 1
tf.reset_default_graph()
sess = tf.Session()
env = gym.make('PongNoFrameskip-v4')

statsFolder = "../experiments/pong_adv_training_0.015/"
dqnModelPath = "../ckpts/dqn/pong_adv_training/0.015/dqn_final.ckpt"
dqn = DQNAgent(env, sess, dqnModelPath)
attackModel = AttackModel(dqn)
stats = testAttacks(dqn, attackModel, attacks, gamesNum, attackProbs, actionProbThr, attDetThr, statsFolder)
