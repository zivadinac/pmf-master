"""Utility functions and variables"""
import tensorflow as tf
import gym
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2grey, gray2rgb
import matplotlib.pyplot as plt
import time
import pickle

def getTerminalStates(env):
    """ Get terminals states of a given env. """
    pS = env.unwrapped.P
    terminalStates = []

    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            psa = pS[state][action]
            for prob in psa:
                if prob[3]:
                    terminalStates.append(prob[1])

    return set(terminalStates)

def oneHot(lentgh, hot):
    """ Generates one-hot encoded vector of a given lentght and givet hot var."""
    vec = np.zeros((1, lentgh))
    vec[0, hot] = 1
    return vec

def extractPolicyFromQ(q):
    """ Extract policy given tabular state-action value function q. """
    return np.argmax(q, 1)

def extractPolicyFromApprox(approx, allStates):
    """ Extract policy given aproximator approx. """
    policy = np.zeros(len(allStates))
    for state in allStates:
        policy[state] = approx(oneHot(len(allStates), state))
    return policy

def epsGreedyProbs(probs, eps):
    """ For given probs create eps-greedy policy. """
    eGP = np.ones(len(probs)) * eps / len(probs)
    eGP[np.argmax(probs)] += 1 - eps
    return eGP

def getMinibatchInds(bs, allInds): 
    mbs = []
    for m in range(int(len(allInds) / bs) + 1):
        inds = allInds[m*bs:m*bs+bs]
        if len(inds) > 0:
            mbs.append(inds)
    return mbs

def preprocess(state):
    s = state.astype(np.float32) / 255.0
    resized = resize(s, (110, 84, 3))
    cropped = resized[17:101, :, :]
    grey = rgb2grey(cropped).reshape((84, 84, 1))
    return (grey - np.mean(grey)).astype(np.float32)

def createAdvDiffFrame(advDiff):
    f = np.zeros((110, 84, 3))
    f[17:101, :, 0] = advDiff
    f[17:101, :, 1] = advDiff
    f[17:101, :, 2] = advDiff
    #return resize(gray2rgb(f), (210, 160, 3)).astype(np.uint8)
    return (resize(f, (210, 160, 3)) * 255).astype(np.uint8)

def pushframe(currentFrames, frame):
    newFrames = np.zeros_like(currentFrames, dtype=np.float32)
    newFrames[0, :, :, 0:3] = currentFrames[:, :, :, 1:]
    newFrames[0, :, :, 3] = frame[:,:,0]
    return newFrames

def save(obj, path):
    f = open(path, "wb")
    pickle.dump(obj, f, protocol=4)
    f.close()

def load(path):
    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()
    return obj
