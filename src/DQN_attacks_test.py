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
    keysSrb = list(map(lambda p: 'p = ' + str(p), probs))
    keysSrb.reverse()
    if attackStats.get('timed') is not None:
        keys.append('timed')
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

def testAttacks(dqn, attackModel, attacksConfig, gamesNum, attackProbs, actionProbThr, attDetThr, saveFolder, stratTimed = True):
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
        if stratTimed:
            attackModel.setActionProbThr(actionProbThr)
            rewards, lengths, meanTime, attNums, attDetStats = testAdversary(gamesNum, dqn, attackModel, attDetThr)

            print(attack + "_timed" + " avgTime = " + str(meanTime) + "s" + " avgReward = " + str(np.mean(rewards)))

            sts["timed"] = {"apThr": attackModel.actionProbThr, "rewards": rewards, "meanReward": np.mean(rewards), "lengths": lengths, "meanLength": np.mean(lengths), "meanTime": meanTime, "meanAttNum": np.mean(attNums), "advDetStats": attDetStats}

        stats[attack] = sts
        utils.save(sts, saveFolder + attack + ".pck")
        savePlotAttackStats(sts, attackProbs, saveFolder)

    return stats

#attacks["deepfool"] = {"nb_candidate": 3, "nb_classes": 6}

gamesNum = 100
attackProbs = [1.0, 0.75, 0.5, 0.25]
actionProbThr = 0.001
attDetThr = 0.9

#testing attacks based on dqn
if True:
    attacks = {}
    attacks["fgsm"] = {"eps": 0.01}
    attacks["momenIt"] = {"eps": 0.015, "eps_iter": 0.7, "nb_iter": 10, "decay_factor": 1.0}
    attacks["spsa"] = {"epsilon": 0.05, "delta": 0.005, "num_steps": 5, "spsa_iters": 5, "spsa_samples": 2, "y_target": 0, "is_targeted": True}#, "early_stop_loss_threshold": -1.}

    tf.reset_default_graph()
    sess = tf.Session()
    env = gym.make('PongNoFrameskip-v4')

    statsFolder = "../experiments/pong/"
    dqn = DQNAgent(env, sess, "../ckpts/dqn/pong_final/dqn_final.ckpt")
    attackModel = AttackModel(dqn)

    stats = testAttacks(dqn, attackModel, attacks, gamesNum, attackProbs, actionProbThr, attDetThr, statsFolder)

    #adv training test
    tf.reset_default_graph()
    sess = tf.Session()
    env = gym.make('PongNoFrameskip-v4')

    statsFolder = "../experiments/pong_adv_training_0.015/"
    dqn = DQNAgent(env, sess, "../ckpts/dqn/pong_adv_training/0.015/dqn_final.ckpt")
    attackModel = AttackModel(dqn)

    stats = testAttacks(dqn, attackModel, attacks, gamesNum, attackProbs, actionProbThr, attDetThr, statsFolder)


#testing attacks based on substitute model
if True:
    attacks = {}
    attacks["fgsm"] = {"eps": 0.015}
    attacks["momenIt"] = {"eps": 0.015, "eps_iter": 0.7, "nb_iter": 10, "decay_factor": 1.0}

    tf.reset_default_graph()
    sess = tf.Session()
    env = gym.make('PongNoFrameskip-v4')

    statsFolder = "../experiments/substitute/"
    dqn = DQNAgent(env, sess, "../ckpts/dqn/pong_final/dqn_final.ckpt")
    sub = SubstituteModel(env, sess, "../ckpts/substitute/final.ckpt")
    attackModel = AttackModel(sub)

    stats = testAttacks(dqn, attackModel, attacks, gamesNum, attackProbs, actionProbThr, attDetThr, statsFolder, stratTimed=False)

    tf.reset_default_graph()
    sess = tf.Session()
    env = gym.make('PongNoFrameskip-v4')

    statsFolder = "../experiments/substitute_adv_training_0.015/"
    dqn = DQNAgent(env, sess, "../ckpts/dqn/pong_adv_training/0.015/dqn_final.ckpt")
    sub = SubstituteModel(env, sess, "../ckpts/substitute/final.ckpt")
    attackModel = AttackModel(sub)

    stats = testAttacks(dqn, attackModel, attacks, gamesNum, attackProbs, actionProbThr, attDetThr, statsFolder, stratTimed=False)
