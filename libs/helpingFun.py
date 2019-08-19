# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import csv
import os

from sklearn.metrics import *
from numpy import genfromtxt


def splitdata(X, Y, Confounders, Noise, percent=[0.8, 0.1, 0.1]):
    np.random.seed(0)
    p = np.random.random(X.shape[0])
    Xtrain = X[p<=percent[0],:]
    Ytrain = Y[p<=percent[0],:]
    Confounders_train = Confounders[p<=percent[0],:]
    Noise_train = Noise[p<=percent[0],:]

    X = X[p>percent[0],:]
    Y = Y[p>percent[0],:]
    Confounders = Confounders[p>percent[0],:]
    Noise = Noise[p>percent[0],:]
    
    np.random.seed(1)
    p = np.random.random(X.shape[0])*(percent[1]+percent[2])
    Xvalid = X[p<=percent[1],:]
    Yvalid = Y[p<=percent[1],:]
    ConfoundersValid = Confounders[p<=percent[1],:]
    NoiseValid = Noise[p<=percent[1],:]

    Xtest  = X[p>percent[1],:]
    Ytest  = Y[p>percent[1],:]
    ConfoundersTest = Confounders[p>=percent[1],:]
    NoiseTest = Noise[p>=percent[1],:]
    return Xtrain, Ytrain, Confounders_train, Noise_train, \
           Xvalid, Yvalid, ConfoundersValid, NoiseValid,\
           Xtest, Ytest, ConfoundersTest, NoiseTest


def reassort(X, Y, idx, yraw, Confounders, X_use, shuffleIdx):

    X = X[shuffleIdx, :]
    Y = Y[shuffleIdx]
    yraw = yraw[shuffleIdx]
    Confounders = Confounders[shuffleIdx]
    X_use = X_use[shuffleIdx, :]

    return X, Y, idx, yraw, Confounders, X_use


def sortByy(X, Y):

    X = X[np.argsort(Y[:,0]), :]
    Y = np.array(sorted(Y))

    return X, Y


def featCausalVec(n_feature, idx):
    # Mark the location of causals for ROC
    feature_causal = np.zeros(n_feature)
    for i in range(len(idx)): 
        feature_causal[idx[i]] = 1
    return feature_causal


def featureRank(feature_weights):
    '''The Rank of every feature'''
    weights_sorted = sorted(abs(feature_weights), reverse = True)
    #rank of each feature
    rank = [list(weights_sorted).index(abs(feature_weights[i])) for i in range(len(weights_sorted))]
    rank_end = len(rank)
    rank_max = max(rank)
    rank = [max(i, (i==rank_max)*rank_end) for i in rank]
    return rank, weights_sorted

    
def getExpNum(i=1):
    while i > 0:
        if not os.path.exists("./save-{}-{}".format(i, 2)):  break    
        i = i+1
    return i
            

def renameSave(i=1, num=2):
    '''Rename the folder `save` when doing multiple experiments'''
    while i > 0:
        if not os.path.exists("./save-{}-{}".format(i, num)): 
            os.rename("./save", "./save-{}-{}".format(i, num))
            break
        i += 1
        
        

def sparseWeight(w, fdr=False):
    from scipy import stats
    w  = np.array(w)
    mu = np.mean(w)
    sigma = np.sqrt(sum((w-mu)**2)/len(list(w)))
    p = stats.norm.cdf(w, loc=mu ,scale=sigma)
    for i, ps in enumerate(p):
        #if 0.05 <= ps <= 0.95: w[i] = 0
        if ps <= 0.90: w[i] = 0  #since given DMM weights are positive
    return w


def predMetrics(idx, feature_weights):
    p = len(list(feature_weights))
    feature_causal = featCausalVec(p, idx)
    auc = roc_auc_score(feature_causal, feature_weights)
    rank, _  = featureRank(feature_weights)
    rank_idx = list(np.array(rank)[idx])
    percent  = sum(np.array(rank)!=p)/float(p)
    return auc, rank_idx, percent
   
     
#(deprecated)
#def predMetrics(feature_causal, feature_weights, weights_sorted=None):
#    
#    auc = roc_auc_score(feature_causal, feature_weights)
#    average_precision = average_precision_score(feature_causal, feature_weights)
#                
#    if weights_sorted==None:
#        weights_sorted = sorted(abs(feature_weights), reverse = True)
#    n = len(weights_sorted)#number of features
#    metricList = []
#    for i in [0.05, 0.1, 0.2]:
#        feature_top = (feature_weights >= weights_sorted[int(i*n)]).astype(int)
#        metricList = [precision_score(feature_causal, feature_top),
#                   recall_score(feature_causal, feature_top),
#                   f1_score(feature_causal, feature_top)]+metricList
#    return auc, average_precision, metricList
    

def transformData(X, Y, Confounders, datatype, noise):
    if datatype== 'continuous':
        Y = Y
    if datatype == 'binary':
        Y = Y - noise
    if datatype == 'confounder':
        Y = noise

    yraw = Y
    Y = yraw + Confounders


    # Change to Binary
    if datatype == 'binary':
        np.random.seed(0)
        for n, i in enumerate(Y):
            Y[n] = np.random.binomial(1, sigmoid(i), 1)[0]

    return X, Y, yraw, Confounders


def getDataForExp(dataset):
    [seed, n, p, noise, CFW, datatype, _, MAF, maxmin, d] = dataset
    params = {'seed': seed, 'n': n, 'p': p, 'd': d, 'noise': noise,
              'confounderWeight': CFW, 'MAF': MAF, 'maxmin': maxmin}

    [_, X, Y, idx, _, _, _, Noise, Confounders, X_use] = getdata('synthetic', params)
    # Transformation
    X, Y, yraw, Confounders = transformData(X, Y, Confounders, datatype, Noise)
    return X, Y, idx, yraw, Confounders, X_use

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def dataLoader(alz1=True):
    if alz1:
        X = np.load('../data/alz_snp.npy').astype(float)
        y = np.load('../data/pheno1.npy').astype(float)
        y = y.reshape([y.shape[0], 1])
        return X, y
    else:
        X = np.load('../data/alz2_snp.npy').astype(float)
        y = np.load('../data/pheno2.npy').astype(float)
        y = y.reshape([y.shape[0], 1])
        return X, y

def loadData():
    X = np.load('../data/X.npy').astype(float)
    y = np.load('../data/Y.npy').astype(int)
    y = y.reshape([y.shape[0], 1])
    return X, y

def loadKinshipMatrix(alz1=True):
    if alz1:
        return np.load('../data/K1.npy').astype(float)
    else:
        return np.load('../data/K2.npy').astype(float)

def calculateKinship(alz1=True):
    X, y = dataLoader(alz1)
    K = np.dot(X, X.T)
    if alz1:
        np.save('../data/K1', K)
    else:
        np.save('../data/K2', K)

if __name__ == '__main__':
    calculateKinship(True)
    calculateKinship(False)
    # X, y = dataLoader(True)
    # print X.shape
    # X, y = dataLoader(False)
    # print X.shape