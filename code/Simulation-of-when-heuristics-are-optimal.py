"""
    Side project: Find heuristic distortions that outperform the ground truth weights
    with logistic outcome


    Testing Tallying + TTB distortions
    with a cross-validation setup (not X + noise)
    logistic version
    adding early and/or late noise


   Varying compensatoriness in the ground truth weights with a sampling process


"""

import os
# keep the names small and concise
path = "/Users/paulaparpart/PycharmProjects/TF_Test/y-loss/noise/Heuristic distortion simulation"
os.chdir(path)
os.getcwd()

sys.path.append(".")
import torch
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch.optim as optim
import inspect
import pandas as pd
import scipy.optimize
from classloss_thres import linearloss, power, g, sigmoid2, stickbreaking
from sklearn.linear_model import LinearRegression

# Sigmoid Model
# 𝛽1 : Controls the curve's steepness, i.e. slope
# 𝛽2 : Slides the curve on the x-axis.
def sigmoid2(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y


def logistic(var):
    p = 1 / (1 + np.exp(-var))  # elementwise devide in python: a/b
    return p



thetas = np.array([0.000000001,0.001, 0.5,1,2,10000000000000000000000])
envs = len(thetas)


## Simulation parameters
iter = 1000
# npred = 10
npred = 4
nsamp = 100
noise = np.linspace(0, 1, 6)
# noise = np.append(noise, [1.5, 2, 3, 4])
exponent = np.linspace(-1, 1, 11)  #(-4, 4, 33)
epsilon = np.linspace(1, 0, 11)
earlynoise = 1
latenoise = 0



# Generate a covariance matrix
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# new version: mix between mine and chris version
mean = np.zeros(npred)
matrixSize = npred
A = np.ones((matrixSize,matrixSize))
levels = np.linspace(0, 0.8, 9)
# constant now
cov = A * levels[3]
np.fill_diagonal(cov, 1)

TTB_loss = np.zeros([iter, len(noise), len(epsilon), envs])
tally_loss = np.zeros([iter, len(noise), len(exponent), envs])
n_included = np.zeros([iter, len(noise), len(epsilon), envs])
n_includedTally = np.zeros([iter, len(noise), len(epsilon), envs])


# compensatory/noncompensatory environments: cycle through ground truth weights
for v in range(envs):

    ## noise loop
    for c in range(0, len(noise)):

        s_noise = 0
        l_noise = 0
        if earlynoise == 1:
            s_noise = noise[c]

        if latenoise == 1:
            l_noise = noise[c]


        print_msg = (
            f'noise: {noise[c]:.2f}')
        print(print_msg)


        for i in range(iter):

            W = stickbreaking(thetas[v], m=npred)
            # this additional line introduces random signs (+/-)
            W = W * np.sign(np.random.normal(0, 1, (npred)))

            X = np.random.multivariate_normal(mean, cov, nsamp)  # Version 1
            # ground truth outcomes
            Y = X @ W

            ## Training data: ground truth + noise
            X_train = X + (np.random.normal(0, 1, size=(nsamp, npred)) * s_noise)  # order with g()
            Y_train = Y + (np.random.normal(0, 1, size=(nsamp)) * l_noise)
            # Fitting: estimate weights in training set
            betas = np.squeeze(np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ Y_train))

            # trim now/or not:
            betas[betas > 1] = 1
            betas[betas < -1] = -1

            # Test data: sample anew just like ground truth X
            X_new = np.random.multivariate_normal(mean, cov, nsamp)
            X_test = X_new  # + (np.random.normal(0, 1, size=(nsamp, npred)) * s_noise)  # with noise now
            Y2_true = X_new @ W


            # TTB: Evaluate classloss (error) at every parameter epsilon: threshold determines n (slope stays constant)
            for p in range(len(epsilon)):
                # exponent k |  epsilon  | slope
                ## TTB: exponent k = 0, 0 < epsilon < 1

                # The problem was previously that the copy of beta changed betas = [0 0000] afte the first iteration with epsilon = 1
                TTB_loss[i, c, p, v], n_included[i, c, p, v] = linearloss([0, epsilon[p], 0], X_test, Y2_true, betas)


            # Tallying: Evaluate classloss (error) at every exponent e, cues = 10
            for e in range(len(exponent)):
                # exponent k |  epsilon  | slope
                tally_loss[i, c, e, v], n_includedTally[i, c, p, v] = linearloss([exponent[e], 0, 0.01], X_test, Y2_true, betas)



# just for plotting
labels = ["~0", ".001", "0.5", "1", "2", "1e+22"]

### TTB: Squeeze the data and visualize: worked :)
fig2, axs = plt.subplots(3, 2, sharex=True, sharey=True)  # normally: (3, 2, sharex=False, sharey=False)
fig2.subplots_adjust(hspace=.25, wspace=.25)
axs = axs.ravel()
colormap = plt.get_cmap('viridis')  # 'PuRd'  'Greys'
colors = [colormap(i) for i in np.linspace(0, 1, len(noise))]  # split colormap into 500 numbers

for v in range(envs):

    # makes it 3-dimensional
    chunk = TTB_loss[:, :, :, v]

    # shape 1000x11 both
    # bool1 = (chunk2[:,c,:] == 1)
    # y = np.mean(chunk[:, c, :][bool1],0) # problem is this turns it to vecotr rather than keeping structure

    # per noise level column
    for c, color in enumerate(colors, start=0):

        # 1000 x 1 x 10
        y = np.mean(chunk[:,c,:], 0) # vector of 11
        err = np.std(chunk[:,c,:], 0)/np.sqrt(TTB_loss.shape[0])
        axs[v].errorbar(epsilon, y, yerr=err , label='Noise level = %.1f' % noise[c], color=color)

        # minimum points:
        #ii = np.where(y == np.min(y))
        axs[v].plot(epsilon[y == np.min(y)][0],  y[y == np.min(y)][0], 'o', markersize=6, color=color)


    # You can use decreasing axes by flipping the normal order of the axis limits: No resorting of data required this way :)
    axs[v].set_xlim(1,0)
    # horizontal lines in every plot
    axs[v].hlines(0, xmin=epsilon[0], xmax=epsilon[len(epsilon) - 1], linestyles='dotted', label='perfect')
    #axs[v].hlines(0.5, xmin=epsilon[0], xmax=epsilon[len(epsilon) - 1], linestyles='dotted', label='chance')
    axs[v].set_ylim(bottom=-0.1)
    axs[v].set_title('TTB $\Theta$ = %s' % (labels[v]))
    axs[v].set_xlabel('epsilon')
    axs[v].set_ylabel('loss (error rate)')
    #axs[v].grid(True)
    axs[v].set_xticks(ticks=epsilon)
    #axs[v].set_xticklabels(xbins)
    #axs[v].legend()
    plt.tight_layout()


    plt.savefig('../TTB_LINEAR_EN_posneg_noncomp_1000env_m%.2f.png' % npred, bbox_inches='tight')
    plt.close()



# Plot the number of features included for TTB: for the same exact conditions
features = np.linspace(0,npred,(npred+1))
fig3, axs = plt.subplots(3, 2, sharex=True, sharey=True)  # normally: (3, 2, sharex=False, sharey=False)
fig3.subplots_adjust(hspace=.25, wspace=.25)
axs = axs.ravel()
colormap = plt.get_cmap('viridis')  # 'PuRd'  'Greys'
colors = [colormap(i) for i in np.linspace(0, 1, len(noise))]  # split colormap into 500 numbers
for v in range(envs):

    # shape: 1000x6x11
    chunk2 = n_included[:, :, :, v]

    # per noise level column
    for c, color in enumerate(colors, start=0):

        # no. of features included graphs
        y2 = np.mean(chunk2[:, c, :], 0)  # vector of 11
        err2 = np.std(chunk2[:, c, :], 0) / np.sqrt(n_included.shape[0])

        axs[v].errorbar(epsilon, y2, yerr=err2, label='Noise level = %.1f' % noise[c], color=color)

    axs[v].set_xlim(1,0)
    axs[v].set_ylim(0,npred)
    axs[v].set_title('TTB $\Theta$ = %s' % (labels[v]))
    axs[v].set_xlabel('epsilon')
    axs[v].set_ylabel('No. of features')
    axs[v].set_xticks(ticks=epsilon)
    axs[v].set_yticks(ticks=features)
    #axs[v].set_yticklabels(np.round(features,0))
    #axs[v].legend()
    plt.tight_layout()


    plt.savefig('../TTB_LINEAR_N_included_thres_noncomp_1000env_m%.2f.png' % npred, bbox_inches='tight')
    plt.close()




### Tallying
fig2, axs = plt.subplots(3, 2, sharex=False, sharey=True)  # normally: (3, 2, sharex=False, sharey=False)
fig2.subplots_adjust(hspace=.25, wspace=.25)
axs = axs.ravel()
colormap = plt.get_cmap('viridis')  # 'PuRd'  'Greys'
colors = [colormap(i) for i in np.linspace(0, 1, len(noise))]  # split colormap into 500 numbers

for v in range(envs):

    # makes it 3-dimensional
    chunk = tally_loss[:, :, :, v]

    # per noise level column
    for c, color in enumerate(colors, start=0):

        # 1000 x 1 x 10
        y = np.mean(chunk[:,c,:], 0) # vector of 10
        err = np.std(chunk[:,c,:], 0)/np.sqrt(tally_loss.shape[0])
        axs[v].errorbar(exponent, y, yerr=err , label='Noise level = %.1f' % noise[c], color=color)

        # minimum points:
        # ii = np.where(y == np.min(y))
        # for i in range(len(ii)):
        #     axs[v].plot(exponent[ii[0]], y[ii[0]], 'o',  markersize= 6, color=color)
        axs[v].plot(exponent[y == np.min(y)][0],  y[y == np.min(y)][0], 'o', markersize=6, color=color)


    # horizontal lines in every plot
    axs[v].hlines(0, xmin = exponent[0], xmax = exponent[len(exponent)-1], linestyles='dotted', label='perfect' )
    #axs[v].hlines(0.5, xmin= exponent[0], xmax= exponent[len(exponent)-1], linestyles='dotted', label='chance')
    axs[v].set_ylim(bottom= -0.1)
    axs[v].set_title('Tallying, $\Theta$ = %s' % (labels[v]))
    axs[v].set_xlabel('compression (k)')
    axs[v].set_ylabel('loss (error rate)')
    #axs[v].grid(True)
    axs[v].set_xticks(ticks=exponent)
    axs[v].set_xticklabels(np.round(exponent,1))
    axs[v].tick_params(axis='x', labelsize=8)
    plt.tight_layout()


    fig2.set_size_inches(10, 8)
    plt.savefig('../Tallying_LINEAR_EN_noncomp_posneg_1000env_m%.2f_more.png' % npred, bbox_inches='tight')
    plt.close()




