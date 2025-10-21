
import torch
import numpy as np
import tensorflow as tf
import torch.optim as optim
import inspect
import torch.nn as nn
import torch.nn.functional as F


## -  Define some functions -
# Sigmoid Model
# 𝛽1 : Controls the curve's steepness, i.e. slope
# 𝛽2 : Slides the curve on the x-axis.
# Power Law Distortion
def power(x, E):
    y = np.sign(x) * (np.absolute(x) ** np.exp(E))
    return y


# g(x, epsilon): | x | < epsilon = 0
def g(x, epsilon):
    # deep copy should not include the reference to betas :)
    B = x.copy()
    # only modify the copy (previously modifying B would modify betas automatically via reference)
    B[np.abs(B) < epsilon] = 0
    return B

# sigmoid2 = logistic() when slope/Beta_1 = 1 and Beta_2 = 0
def sigmoid2(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y


def logistic(var):
    p = 1 / (1 + np.exp(-var))  # elementwise devide in python: a/b
    return p


# Split the data into mini-batches:
def merge(list1, list2):
    merged_list = [[list1[i], list2[i]] for i in range(0, len(list1))]
    return merged_list


