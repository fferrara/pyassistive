# -*- coding: utf-8 -*-
"""
Metodi vari per valutazione performance
"""

import numpy as np

def get_confusion_matrix(label, output, classes):
    if len(label) != len(output):
        label = label[0:len(output)]

    matrix = np.zeros((classes, classes))
    for (l, o) in zip(label, output):
        if o >= 0:
            matrix[l, o] += 1

    return matrix

def get_accuracy(matrix):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError

    diag = np.sum(matrix * np.eye(len(matrix)))
    return diag / np.sum(matrix)

def get_cohen_k(matrix):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError

    actuals = np.sum(matrix, axis=0)
    predicted = np.sum(matrix, axis=1)
    total = np.sum(matrix)
    Pe =  np.sum(actuals / total * predicted / total)
    Po = get_accuracy(matrix)

    return (Po - Pe) / (1 - Pe)


def get_ITR(classes, accuracy, trial):
    """
    Standard ITR formula
    """
    np.seterr(divide='ignore', invalid='ignore')
    itr = np.log2(classes) + np.nan_to_num(accuracy * np.log2(accuracy)) + np.nan_to_num((1 - accuracy) * np.log2((1 - accuracy) / (classes - 1)))
    np.seterr(divide='warn', invalid='warn')
    return (1/trial) * itr


def get_utility(devices, accuracy, trial):
    """
    See articles.
    """
    dsts = np.array([0,1,2,3,2,1])
    if accuracy < 0.5:
        return 0
    else:
        return np.mean(((2*accuracy - 1)*np.log2(devices)) / ((dsts+1) * trial))