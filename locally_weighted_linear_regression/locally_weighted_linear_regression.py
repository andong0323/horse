#!/usr/bin/env python
#coding:utf-8

from math import exp
import numpy as np
import matplotlib.pyplot as plt
from standard_linear_regression import load_data, get_corrcoef


def lwlr(x, X, Y, k):
    m = X.shape[0]
    W = np.matrix(np.zeros((m, m)))
    for i in range(m):
        xi = np.array(X[i][0])
        x = np.array(x)
        W[i, i] = exp

#对某一点计算估计值
def lwlr_bak(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        weights[i, i] = exp(diffMat * diffMat.T / (-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print('This Matrix is singular, cannot be inverse')
        return
    theta = xTx.T * (xMat.T * (weights * yMat))
    return testPoint * theta

#对所有点计算估计值
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat
