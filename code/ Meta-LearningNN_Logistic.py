import sys

sys.modules[__name__].__dict__.clear()


"""
    Meta-learning neural network architecture for the Logistic Case (binary outcome) 
    
    The Neural net is given input data 
        - with varying statistical parameters such as noise levels, size, number of predictors, covariance level etc.
    The network learns a set of weights that best generalizes from training to test data
        - this set of weights can represent a heuristic (ML regularizers) or other strategy 
        
    
    What does this script do? Simulation that uses the network to predict y_hat with logistic y outcome
    loss function is binary cross entropy loss (at dataset level) 
     
    npred can be set at start 
    
    Measure outcome as: 
    1) Multiplot of y_loss plot (see how fast converges) 
    3) Multiplot of weight clouds (w_hat over ground truth) just for 1 batch for epoch = 200
    
 
    Hyperparameters of optimal network - may need to be adjusted:
    epochs = 100  
    ndata = 1000,000
    batch size b = 500 
    #training batches = 1800
    LR = 0.001 
    noise = Norm(0,1) * [0.001, 0.01, 0.1, 0.2, 0.5, 0.9]

"""

import os
# keep the names small and concise
path = "/Users/paulaparpart/PycharmProjects/TF_Test/y-loss/noise/Grid Logistic MLP"
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
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
import torch.nn as nn
import torch.nn.functional as F


# Parameters outside the loop
iter = 100  # epochs
b = 500  ## Mini-batch training
L_param = 0.001
decay_param = 0
nhid = 200
# CAN ADJUST: TEST 10
npred = 4  #2,4,10
nsamp = 20
noise = np.linspace(0, 1, 6) # matched to grid search
earlynoise = 1
latenoise = 0
# store these variables below too in plots and data names
negative = 1

# for covariance: for now we test 1 level of cov (0) to see effect just of noise
mean = np.zeros(npred)
matrixSize = npred
A = np.ones((matrixSize,matrixSize))
levels = np.linspace(0, 0.8, 9)
## MODIFY
level = levels[8]
cov = A * levels[8]
np.fill_diagonal(cov, 1)
# for now: change later
slope = 1 # 0.1   1 to match nn.sigmoid() slope in y_hat_test



## -  Define some functions -

# My own weight initialization function:
def he_initialize(shape):
    """
    Kaiming He normalization: sqrt(2 / fan_in)
    better for ReLu activation fn
    """
    fan_in = shape[1]  # 2nd tensor dimension (size[l-1])
    # eg shape would be torch.Size([200, 220]
    w = torch.randn(shape) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w


def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.kaiming_normal_(m, mode='fan_in', nonlinearity='relu')
        m.weight.data = he_initialize(m.weight.data.size())


# for testing:
# D = inputs_test[:, 0:(nsamp * npred)]
# weights = torch.randn(500, 2)

# layer activation function
def transform(weights, D):
    """
    Custom activation function: output * x_test = y_hat
    weights = output = 500xnpred
    D = x_test = 500x40 usually
        - sometimes D may be inputs_train, then use only D = D[:, 0:(D.shape[1] - 20)] = 500x40
    out = 500x20  (y_hat)

    now adjusted to general case: Any number of predictors npred
    """

    pred = weights.shape[1]  # e.g., if this is 3 then D.view(500, 20, 3) from original D = 500x60 etc.

    # reshape D = 500x40 into D = 500x20x2
    x = D.view(500, 20,
               pred)  # x[1, 0, :] shows that the first row of the first tensor has grouped the entries of D[1, :] by pairs too, taking two adjacent entries and putting them into a row (= the reverse of the flatten() operation)
    W = weights.view(500, pred, 1)  # weights[0,:] and W[0,:,:] shows reshape worked correctly
    # batch-multiply two 3D tensors :)   500x20x2 * 500x2x1 = 500x20x1
    out = torch.bmm(x, W)  # torch.Size([500, 20, 1])
    # now reshape out back into 500x20 for y_hat
    out = out.view(500, 20)  # out[0, :, :] and T[0, :] are equivalent so it worked
    return out

# Split the data into mini-batches:
def merge(list1, list2):
    merged_list = [[list1[i], list2[i]] for i in range(0, len(list1))]
    return merged_list


# Sigmoid Model
# 𝛽1 : Controls the curve's steepness, i.e. slope
# 𝛽2 : Slides the curve on the x-axis.
def sigmoid2(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y

def logistic(var):
    p = 1 / (1 + np.exp(-var))  # elementwise devide in python: a/b
    return p
# sigmoid2 = logistic() when slope/Beta_1 = 1 and Beta_2 = 0




# Cycle through noise - parameters:
for c in range(0, 1):  # len(noise)

    s_noise = 0
    l_noise = 0
    if earlynoise == 1:
        s_noise = noise[c]
    if latenoise == 1:
        l_noise = noise[c]

    # Create trainset and evaluation set both with 900000 data points, i.e, 1800 batches
    # Y-hat: Create equal-size trainset and validset as part of the same loop: Same generating W
    Tdata = 1000000
    ndata = 900000

    labels = torch.empty(ndata, npred)
    ground_truth = torch.empty(ndata, npred)

    inputs = torch.empty(ndata, (nsamp * npred) + nsamp)
    labelsv = torch.empty(ndata, npred)
    inputsv = torch.empty(ndata, (nsamp * npred) + nsamp)  # 500 x 220
    for k in range(0, ndata):

        # Generative process matching grid search
        # ground truth
        W = np.random.rand(npred)  # for now try weights between 0 -1 first, then negative too
        W = W * np.sign(np.random.normal(0, 1, (npred)))
        # W = W * np.sign(np.random.normal(0, 1, (npred)))
        X = np.random.multivariate_normal(mean, cov, nsamp)
        Y = X @ W


        X_train =  X + (np.random.normal(0, 1, size=(nsamp, npred)) * s_noise) # what was D1 before
        Y_train = Y + (np.random.normal(0, 1, size=(nsamp)) * l_noise)
        #P_train = sigmoid2(Y_train, (1 / slope), 0)  + (np.random.normal(0, 1, size=(nsamp)) * l_noise)  # noise is added after prob as in logistic() version
        betas = np.squeeze(np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ Y_train))

        # GLM
        # Y_samp = np.random.rand(nsamp) < P_train
        # glm_binom = sm.GLM(Y_samp, X_train, family=sm.families.Binomial())
        # res = glm_binom.fit(maxiter=1000)
        # betas = res.params   # might need to be normalized to 0-1
        # Fitting: estimate weights in training set, with outcome = probability
        # glm_gamma = sm.GLM(P_train, X_train, family=sm.families.Gamma(sm.families.links.log()))
        # glm_results = glm_gamma.fit()
        # betas = glm_results.params

        # # labels = emp. weights now
        labels[k, :] = torch.from_numpy(betas.flatten())
        # store ground truth nonetheless too
        ground_truth[k, :] = torch.from_numpy(W.flatten())

        #print(W, betas)

        ## Test data:
        X_new = np.random.multivariate_normal(mean, cov, nsamp)
        X_test = X_new + (np.random.normal(0, 1, size=(nsamp, npred)) * s_noise)  # previously D2
        Y_test = X_new @ W      # previously Y2
        P_test = sigmoid2(Y_test, Beta_1=(1 / slope), Beta_2=0)  # sigmoid2 = logistic() when slope/Beta_1 = 1 and Beta_2 = 0

        # trainset in grid search
        # columns 0:200, put in Data D flattened, and 200:220 it is Y flattened
        inputs[k, 0:(nsamp * npred)] = torch.from_numpy(X_train.flatten())
        inputs[k, (nsamp * npred):((nsamp * npred) + nsamp)] = torch.from_numpy(Y_train.flatten())

        # testset in grid search
        # columns 0:200, put in Data D flattened, and 200:220 it is Y flattened
        inputsv[k, 0:(nsamp * npred)] = torch.from_numpy(X_test.flatten())
        inputsv[k, (nsamp * npred):((nsamp * npred) + nsamp)] = torch.from_numpy(P_test.flatten())


    # trainset is 1 list with 2 tensors
    trainset = [inputs, labels]  # data: Xy and  empirical weights (labels)

    # Test data for network: actually labels should be empirical weights in the testset, but it is never used
    validset = [inputsv, labels]


    # save both torches for future testing
    torch.save(trainset, 'Trainset_TESTING_LOGISTIC_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_negative%.0f_slope%.0f.pt' % (
        earlynoise, latenoise, noise[c], npred, negative, slope))

    # trainset = torch.load('Trainset_TESTING_LOGISTIC_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_negative%.0f.pt' % (
    # earlynoise, latenoise, noise[c], npred, negative))

    torch.save(validset, 'Validset_TESTING_LOGISTIC_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_negative%.0f_slope%.0f.pt' % (
        earlynoise, latenoise, noise[c], npred, negative, slope))
    #
    # validset = torch.load('Validset_TESTING_LOGISTIC_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_negative%.0f.pt' % (
    #     earlynoise, latenoise, noise[c], npred, negative))
    #


    ## Generate 1st and 2nd Testset for Eval() step  - Pairs ƒ- same generating W
    tdata = ndata  # 10%
    labelst = torch.empty(tdata, npred)
    inputst = torch.empty(tdata, (nsamp * npred) + nsamp)  # 500 x 220
    labelst2 = torch.empty(tdata, npred)
    inputst2 = torch.empty(tdata, (nsamp * npred) + nsamp)  # 500 x 220
    for m in range(0, tdata):

        W = np.random.rand(npred)  # for now try weights between 0 -1 first, then negative too
        W = W * np.sign(np.random.normal(0, 1, (npred)))
        # W = W * np.sign(np.random.normal(0, 1, (npred)))
        X = np.random.multivariate_normal(mean, cov, nsamp)
        Y = X @ W

        X_train = X + (np.random.normal(0, 1, size=(nsamp, npred)) * s_noise)  # what was D1 before
        Y_train = Y + (np.random.normal(0, 1, size=(nsamp)) * l_noise)
        #P_train = sigmoid2(Y_train, (1 / slope), 0) + (np.random.normal(0, 1, size=(nsamp)) * l_noise)  # noise is added after prob as in logistic() version

        ## Test data:
        X_new = np.random.multivariate_normal(mean, cov, nsamp)
        X_test = X_new + (np.random.normal(0, 1, size=(nsamp, npred)) * s_noise)  # previously D2
        Y_test = X_new @ W  # previously Y2
        P_test = sigmoid2(Y_test, Beta_1=(1 / slope), Beta_2=0)  # infliction = 0, inverse slope = 0.1, slope = 10, height = 1, minnie = 0

        inputst[m, 0:(nsamp * npred)] = torch.from_numpy(X_train.flatten())
        inputst[m, (nsamp * npred):((nsamp * npred) + nsamp)] = torch.from_numpy(Y_train.flatten())
        # columns 0:200, put in Data D flattened, and 200:220 it is Y flattened
        inputst2[m, 0:(nsamp * npred)] = torch.from_numpy(X_test.flatten())
        inputst2[m, (nsamp * npred):((nsamp * npred) + nsamp)] = torch.from_numpy(P_test.flatten())

    testset1 = [inputst]  # for training inside the network
    testset2 = [inputst2]

    del inputs, labels, inputsv, labelsv, k, m, labelst, labelst2, inputst, inputst2


    ## Training Step x 2:
    # Split the trainset and testset into mini-batches
    input_split = torch.split(trainset[0], b, dim=0)
    labels_split = torch.split(trainset[1], b, dim=0)
    Trainloader = merge(input_split, labels_split)
    ## Mini-batches for Validation:
    input_split = torch.split(validset[0], b, dim=0)
    labels_split = torch.split(validset[1], b, dim=0)
    Validloader = merge(input_split, labels_split)

    # split the ground truth in same way?
    # trainset[1] = labels  like ground_truth
    w_split = torch.split(ground_truth, b, dim=0)
    GT_loader = w_split

    ## Evaluation Step:
    ## Mini-batches for Train at Eval:
    input_split = torch.split(testset1[0], b, dim=0)
    Testloader1 = input_split
    ## Mini-batches for Test at Eval:
    input_split = torch.split(testset2[0], b, dim=0)
    Testloader2 = input_split

    del trainset, validset, testset1, testset2


    # 2. Define a Deep - Feedforward Network
    D_in = (nsamp * npred) + nsamp  # ncols of a batch (e.g., 60) or (20 * 6) + 20 = 140
    D_out = npred

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # input to H1
            self.fc1 = torch.nn.Linear(D_in, 200, bias=False)
            # relu activation
            self.relu = torch.nn.ReLU()
            # H1 to H2
            self.fc2 = torch.nn.Linear(200, 200, bias=False)
            # H2 to output Y
            # instead of 10 it is npred
            self.fc3 = nn.Linear(200, D_out, bias=False)  # linear output activation function: creates 500x10 fY output matrix
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()

        def forward(self, x, x_test):
            # all the self. functions below have to be defined in _init_
            hidden1 = self.fc1(x)
            relu = self.relu(hidden1)
            hidden2 = self.fc2(relu)
            relu2 = self.relu(hidden2)
            output = self.fc3(relu2)
            output = self.tanh(output) # this allows for weights [-1,+1] and relu before is ok
            #output = self.sigmoid(output)
            y_hat = transform(output, x_test)  # takes output from last layer and testdata to do: W * X_test = y_hat_test     = 500x20
            y_hat = self.sigmoid(y_hat)  # 500x20

            return y_hat, output

    net = Net()

    net.apply(init_weights)



    # 3. Train the network

    # Define a Loss function and optimizer
    # loss='binary_crossentropy' better for logistic version with 2 probabilities
    # def loss_fn(input, target):
    #     """ manual euclidean loss function  """
    #     # loss_row = torch.abs(input - target).mean(1)  # .mean(1) gives row-means (MSE), .sum(1) gives row-sums (SSE)
    #     loss_row = ((input - target) ** 2).mean(1)  # MSE
    #     return loss_row
    # Manual Binary Cross Entropy
    # def bce(x,y):
    #     # x,y could matrices or vectors
    #     # when x,y are vectors, this returns 20 elements
    #     # so if i apply it row-wise it would give 500x20 unless i mean it
    #     l_n = -1 * (y * x.log() + (1 - y) * (1 - x).log())
    #
    #     ##  as if:        P2_true         P2_heuristic.log()
    #     #l_n = -1 * (y_test[i,:] * y_hat_test[i,:].log() +   (1 - y_test[i,:]) * (1 - y_hat_test[i,:]).log())
    #
    #     l_mean = l_n.mean()  # apply mean just per row
    #     return l_mean
    #
    # def loss_fn(x, y):
    #     """ manual BCELoss row-wise (pytorch can only do mean across all elements, or returns 500x20 outputs)  """
    #     # x = y_hat_test  # 500x20 matrix
    #     # y = y_test
    #     loss_row = torch.empty(x.shape[0])
    #     ## apply the bce function by row to the matrices
    #     for i in range(x.shape[0]):
    #         loss_row[i] = bce(y_hat_test[i,:], y_test[i,:])
    #     # tf.map_fn(bce, (y_hat_test))  # cannot take two inputs into bce function and apply to them differently
    #     return loss_row
    #

    # loss_fn = nn.BCELoss()  # mean
    loss_fn = nn.BCELoss(reduction='none')  # 500x20, reduced more afterwards

    optimizer = optim.SGD(net.parameters(), lr=L_param, weight_decay=decay_param)
    avg_train_loss = np.zeros(iter)
    avg_valid_loss = np.zeros(iter)
    avg_w_loss = np.zeros(iter)
    batch_loss = np.zeros([iter, len(Trainloader)])  # 10,000 x 18 matrix, change data type? or do i need to specify?
    batch_loss_w = np.zeros([iter, len(Trainloader)])
    batch_loss_valid = np.zeros([iter, len(Validloader)])  # 10,000 x 18 matrix, change data type? or do i need to specify?
    snapweights = torch.empty(iter, b, npred)


    #iter = 20

    for epoch in range(iter):  # loop over the dataset multiple times

        ###################
        # train the model: mini-batch
        ###################
        net.train()  # prep model for training
        for i, data in enumerate(Trainloader):  # i is counter, data is content
            # get the inputs; data is a list of [inputs, labels]
            inputs_train, labels_train = Trainloader[i]  # inputs_train is 500x60 i.e, 500 x ((nsamp x npred) + nsamp). labels is 500x2
            inputs_test, labels_test = Validloader[i]
            # zero the parameter gradients
            optimizer.zero_grad()
            # inputs train contains x and y (500x60), while inputs_test only contains x_test (500x40)
            y_hat_test, weights = net(inputs_train, inputs_test[:, 0:(nsamp * npred)])
            y_test = inputs_test[:, (nsamp * npred):((nsamp * npred) + nsamp)]  # y_test.shape is 500x20, so y_hat_test should also be 500x20
            loss = loss_fn(y_hat_test,y_test)  # y_test = P2_true as in grid search; y_hat_test = P2_heuristic but with diff sigmoid slope= 1 from nn.sigmoid()
            loss = loss.mean(1)  # Row-means ([500])

            # loss needs to be shape 500
            v = torch.ones(b, dtype=torch.float)
            loss.backward(v)
            optimizer.step()
            batch_loss[epoch, i] = torch.mean(loss).item()  # average distance of 20 y's to y_hat, averaged per batch, across 500 datasets

            #print(i, batch_loss[epoch, i])

            # Storing weights from the first batch (i = 0) per epoch e, so we can track their development over time later, for the same 500 datasets
            if i == 0:
                snapweights[epoch, :, :] = weights.detach()  # 200x500x2

        avg_train_loss[epoch] = np.mean(batch_loss[epoch, :])


        ######################
        ## once you know the validation loss works, comment below out!
        # validate the model: Evaluation on independent testsets (same distribution) #
        ######################
          # net.eval()
        # # with 100.000 testset, these are 200 batch runs
        # with torch.no_grad():
        #     # cycle through test batches too
        #     for l, data in enumerate(Testloader1):
        #
        #         inputs_train, labels_train = data  # Testloader1[l]
        #         inputs_test, labels_test = Testloader2[l]  # Testloader2 needs to have same dimensions as Testloader
        #         # Network needs 2 inputs now
        #         y_hat_test_eval, weights_eval = net(inputs_train, inputs_test[:, 0:(nsamp * npred)])
        #         y_test_eval = inputs_test[:, (nsamp * npred):((nsamp * npred) + nsamp)]
        #         loss2 = loss_fn(y_hat_test_eval, y_test_eval)
        #         batch_loss_valid[epoch, l] = torch.mean(loss2).item()  # gets the average test loss in 200 different batches with those net parameters from epoch
        #
        #
        # # calculate average loss over an epoch: mean per epoch is still going to be in the 400s if the sum(Loss) is taken above for a batch (batch size = 500 makes it be around 400)
        avg_valid_loss[epoch] = np.mean(batch_loss_valid[epoch, :])

        # print the current train and val loss
        epoch_len = len(str(iter))
        print_msg = (f'[{epoch:>{epoch_len}}/{iter:>{epoch_len}}] ' +
                     f'train_loss: {avg_train_loss[epoch]:.5f} ' +
                     f'valid_loss: {avg_valid_loss[epoch]:.5f}')

        print(print_msg)


    print('Finished Training and Testing')




    inputs_train, emp_weights = Trainloader[0]
    # snapweights are also from Trainloader[0] but stored throughout time (epochs)
    snapweights = snapweights[(iter -1), :, :] # just need the snapweights (color cloud) at epoch = 200: dimensions 500x2
    #result =  torch.cat((snapweights, emp_weights, GT_loader[0]), 1)
    result = torch.cat((snapweights, emp_weights), 1)

    #result = torch.cat((snapweights, emp_weights), 1)
    torch.save(result,'Logistic_snapweights_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_b%.0f_epochs%.0f_BCE_negweights%.0f_cov%.2f.pt' % (earlynoise, latenoise, noise[c], npred, b,  round(iter), negative, level))


    # 5. Save the data in a data.frame
    import pandas as pd
    df = pd.DataFrame(dict(epoch=range(iter), avg_loss_train=avg_train_loss, avg_loss_test=avg_valid_loss,
                           avg_w_loss = avg_w_loss,
                           mean_loss_train=avg_train_loss.mean(), mean_loss_test=avg_valid_loss.mean()))
    df.to_csv("Logistic_snapweights_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_output_data_BCE_negweights%.0f_cov%.2f.csv" % (earlynoise, latenoise, noise[c], npred, negative, level),  index=False)



print('Finished Noise simulations')





## After simulation, Multi-plot y-loss:
path = "/Users/paulaparpart/PycharmProjects/TF_Test/y-loss/noise"
os.chdir(path)
os.getcwd()

fig2, axs = plt.subplots(3, 2, sharex=False, sharey=True)   # normally: (3, 2, sharex=False, sharey=False)
fig2.subplots_adjust(hspace=.25, wspace=.25)
axs = axs.ravel()
fig2.suptitle('Training and Testing Loss', fontsize=16)

for c in range(0, len(noise)):


    # df = pd.read_csv("Logistic_snapweights_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_output_data_BCE_negweights%.0f.csv" % (
    # earlynoise, latenoise, noise[c], npred, negative), sep=",", header=0)

    df = pd.read_csv("Logistic_snapweights_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_output_data_BCE_negweights%.0f_cov%.0f.csv" % (earlynoise, latenoise, noise[c], npred, negative, level), sep  = ",", header = 0)
    axs[c].plot(range(1, df.shape[0] + 1), df[['avg_loss_train']], 'g', label='Train loss (avg. batch)')
    #axs[c].plot(range(1, df.shape[0] + 1), df[['avg_loss_test']], 'b', label='Validation loss (avg. batch)')
    axs[c].set_title('Noise level = %.3f' % noise[c])
    axs[c].set_xlabel('Epochs')
    axs[c].set_ylabel('Loss')
    #axs[c].set_ylim(bottom=0)
    axs[c].grid(True)
    axs[c].legend()
    plt.tight_layout()


    fig2.set_size_inches(15, 12)
    plt.savefig('../noise/LogisticGrid_BCE_Noise_yloss_EN%.0f_LN%.0f_npred%.0f_batch%.0f_epochs%.0f_samey_negweights%.0f_cov%.2f.pdf' % (earlynoise, latenoise, npred, b, round(iter), negative, level))
    plt.close()



## Single plot W cloud (noise = 0): weight recovery
c = 0
cloud = torch.load('Logistic_snapweights_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_b%.0f_epochs%.0f_BCE_negweights%.0f.pt' % (earlynoise, latenoise, noise[c], npred, b,  round(iter), negative))


#cloud = torch.load('Logistic_snapweights_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_b%.0f_epochs%.0f_BCE_negweights%.0f_cov%.2f.pt' % (earlynoise, latenoise, noise[c], npred, b,  round(iter), negative, level))
#result = torch.cat((snapweights, emp_weights), 1)

# xdata = empirical weights, ydata = NN optimal weights
#xdata, ydata, gt = (cloud[:, npred: (npred * 2)], cloud[:, 0:npred], cloud[:, (npred * 2):cloud.shape[1]])

xdata, ydata = (cloud[:, npred: (npred * 2)], cloud[:, 0:npred])
xdata = xdata.numpy()
ydata = ydata.numpy()

number = b
# colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired
colormap = plt.get_cmap('tab20')  # 'PuRd'  'Greys'  'hsv'
colors = [colormap(i) for i in np.linspace(0, 1, number)]  # split colormap into 500 numbers

for i, color in enumerate(colors, start=0):  # 500

    #if (i + 1) % 5 == 0:  # = 100 datasets (still visible and not too few)
    plt.plot(xdata[i,], ydata[i,], 'o', markersize=4, color=color)  # 1 color per dataset
        #plt.plot(gt[i,], ydata[i,], 'o', markersize=4, color=color)  # 1 color per dataset

plt.title('Distribution of $\hat w$ and $w$ (across 500 datasets)')
plt.xlabel('Empirical weights ($w_{train}$)')
plt.ylabel('$\hat w$')
plt.grid(True)
plt.tight_layout()
lims = [
    np.min([plt.xlim(), plt.ylim()]),  # min of both axes
    np.max([plt.xlim(), plt.ylim()]),  # max of both axes
]
# now plot both limits against each other
plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
plt.show()

plt.savefig('../noise/Logistic_Weightrecovery_negweights%.0f_npred_%.0f.png' % (negative, npred))
plt.close()



## Power Law fitting: joint fit

# Power Law Distortion
def power(x, E, epsilon):

    B = x.copy()
    B[np.abs(B) < epsilon] = 0
    y = np.sign(B) * (np.absolute(B)**np.exp(E))
    return y


# Power Law Distortion
def power2(x, E):

    # assuming epsilon = 0 (no step function)
    y = np.sign(x) * (np.absolute(x)**np.exp(E))
    return y

# 1 - parameter fit: k
#noise = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9])

number = b
# colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired
colormap = plt.get_cmap('tab20')  # 'PuRd'  'Greys'  'hsv'
colors = [colormap(i) for i in np.linspace(0, 1, number)]  # split colormap into 500 numbers

fig2, axs = plt.subplots(2,2, sharex=False, sharey=True)   # normally: (3, 2, sharex=False, sharey=False)
fig2.subplots_adjust(hspace=.25, wspace=.25)
axs = axs.ravel()
fig2.suptitle('Distribution of $\hat w$ and $w$', fontsize=12)
plt.tight_layout() # now it makes sure size is right before plotting


for c in range(0, len(noise)):
    # read in data:
    cloud = torch.load(
        'Logistic_snapweights_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_b%.0f_epochs%.0f_BCE_negweights%.0f_cov%.2f.pt' % (
        earlynoise, latenoise, noise[c], npred, b, round(iter), negative, level))

    #cloud = torch.load('Linear_snapweights_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_b%.0f_epochs%.0f.pt' % (earlynoise, latenoise, noise[c], npred, b, round(iter)))
    xdata, ydata, gt = (cloud[:, npred: (npred * 2)], cloud[:, 0:npred], cloud[:, (npred * 2):cloud.shape[1]])
    xdata = xdata.numpy()
    ydata = ydata.numpy() # 500x6
    ## Trim before ravel: -1, +1 otherwise sigmoids fit out of range
    xdata[xdata < -1] = -1
    xdata[xdata > 1] = 1
    X = xdata.ravel() # 3000,
    Y = ydata.ravel()

    # sigmoid predictions for Plotting: based on 3000 data points
    #popts, pcovs = curve_fit(sigmoid2, xdata, ydata)

    # Least squares: cont. function; 1 parameter k
    popts, pcovs = curve_fit(power2, X, Y)

    # Plot fitted curve into the plot
    x_s = np.linspace(-1, 1, 1000)
    #x_s = np.linspace(np.min(X),np.max(X), 1000)
    y_s = power2(x_s, *popts)

    # this is like asking, does collectively across 500 datasets (a batch) a sigmoid fit better?
    for i, color in enumerate(colors, start=0):  # 500
        axs[c].plot(xdata[i,], ydata[i,], 'o', markersize=4, color=color)  # 1 color per dataset

    # Sigmoidal fit:
    axs[c].plot(x_s, y_s, linewidth=3.0, label=r'Power distortion fit: k =%4.2f' % (popts[0]), color='darkblue')  # sigmoidal fit
    #axs[c].plot(x_s, y_s, linewidth=2.0, label=r'Sigmoidal fit: b1 =%4.2f, b2=%4.2f' % (popts[0], popts[1]),color='darkblue')  # sigmoidal fit
    #plt.tight_layout()
    axs[c].set_title('Noise level = %.3f' % noise[c])
    axs[c].set_xlabel('Empirical weights ($w_{train}$)')
    axs[c].set_ylabel('$\hat w$')
    axs[c].grid(True)
    axs[c].legend(loc='best')
    plt.show()


    fig2.set_size_inches(14, 10)
    plt.savefig('../noise/Logistic_Multiplot_Curvefits_w_hat_by_colour_EN%.0f_LN%.0f_pred%.0f_b%.0f_negative%.0f_1params_cov%.2f.pdf' % (earlynoise, latenoise,  npred, b, negative, level))
    plt.close()










# Sigmoid Model
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y

## Sigmoidal curve fits:
fig3, axs = plt.subplots(3, 2, sharex=False, sharey=True)   # normally: (3, 2, sharex=False, sharey=False)
fig3.subplots_adjust(hspace=.25, wspace=.25)
axs = axs.ravel()
fig3.suptitle('Distribution of $\hat w$ and $w$', fontsize=12)
plt.tight_layout() # now it makes sure size is right before plotting

number = b
#colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired
colormap = plt.get_cmap('tab20')   #'PuRd'  'Greys'  'hsv'
colors = [colormap(i) for i in np.linspace(0, 1, number)] # split colormap into 500 numbers

for c in range(0, len(noise)):

    # read in data:
    cloud = torch.load('Logistic_snapweights_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_b%.0f_epochs%.0f.pt' % (earlynoise, latenoise, noise[c], npred, b, round(iter)))

    # xdata = empirical weights, ydata = NN optimal weights
    xdata, ydata = (cloud[:, npred : (npred*2)], cloud[:, 0:npred])
    xdata = xdata.numpy()
    ydata = ydata.numpy()

    ## Now trimming empirical weights:
    xdata[xdata < 0] = 0
    xdata[xdata > 1] = 1

    params = np.empty([b, 2])
    for i, color in enumerate(colors, start=0): #500

        # but actually fit each of the  500 datasets and get the average steepness parameter (Beta)
        # popts[0] = steepness parameter, popts[1] = slides curves on x-axis

        # sigmoid predictions for Plotting
        popts, pcovs = curve_fit(sigmoid, xdata[i, :], ydata[i,:], maxfev = 5000)

        # to improve fit: p0 = [1, 0, 1], maxfev = 5000
        # if # there is an error
        #     popts = np.array([np.nan, np.nan])
        # else:
        #     params[i,] = popts

        ## Store them each and then create a histogram of the steepness parameters across the 500 datasets.
        params[i, ] = popts

        #if i < 170:
        #if (i + 1) % 25 == 0:  # = 20 datasets (still visible and not too few)
        if (i+1)%10==0:  # = 50 datasets (still visible and not too few)

            axs[c].plot(xdata[i, ], ydata[i, ], 'o', markersize=4, color=color)  # 1 color per dataset
            # Sigmoidal fit:
            # range of actual emp data
            # x_s = np.linspace(min(xdata[i, ]),max(xdata[i, ]), 1000)
            x_s = np.linspace(0, 1, 1000)  #what resolution is good?
            y_s = sigmoid(x_s, *params[i,])
            axs[c].plot(x_s, y_s, linewidth=2.0, color = color)  # is same as dots now from cmap

    #plt.tight_layout()
    axs[c].set_title('Noise level = %.1f' % noise[c])
    axs[c].set_xlabel('Empirical weights ($w_{train}$)')
    axs[c].set_ylabel('$\hat w$')
    axs[c].grid(True)
    plt.show()


    fig3.set_size_inches(14, 10)
    plt.savefig('../noise/Logistic_Grid_Individual_Curvefits_w_hat_by_colour_trimmed_%.0fnpred.pdf' % npred)
    plt.close()



## Histogram with params[:, 0]
fig3, axs = plt.subplots(3, 2, sharex=False, sharey=False)   # normally: (3, 2, sharex=False, sharey=False)
fig3.subplots_adjust(hspace=.25, wspace=.25)
axs = axs.ravel()
fig3.suptitle('Fitted sigmoidal slopes', fontsize=12)
plt.tight_layout()

number = b
#colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired
colormap = plt.get_cmap('tab20')   #'PuRd'  'Greys'  'hsv'
colors = [colormap(i) for i in np.linspace(0, 1, number)] # split colormap into 500 numbers

for c in range(0, len(noise)):

    # read in data:
    cloud = torch.load('Logistic_snapweights_EN%.0f_LN%.0f_Noise%.3f_pred%.0f_b%.0f_epochs%.0f.pt' % (earlynoise, latenoise, noise[c], npred, b, round(iter)))

    # xdata = empirical weights, ydata = NN optimal weights
    xdata, ydata = (cloud[:, npred: (npred * 2)], cloud[:, 0:npred])
    xdata = xdata.numpy()
    ydata = ydata.numpy()

    ## Now trimming empirical weights:
    xdata[xdata < 0] = 0
    xdata[xdata > 1] = 1

    params = np.empty([b, 2])
    for i, color in enumerate(colors, start=0):  # 500

        # popts[0] = steepness parameter, popts[1] = slides curves on x-axis
        popts, pcovs = curve_fit(sigmoid, xdata[i, :], ydata[i, :], maxfev=5000)
        params[i,] = popts

    # Plot all of the 5000 slopes (might as well)  per noise condition
    axs[c].hist(x= params[:, 0], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    axs[c].grid(axis='y', alpha=0.75)
    axs[c].set_xlabel('Slope Value')
    axs[c].set_ylabel('Frequency')
    axs[c].set_title('Noise level = %.1f' % noise[c])
    axs[c].set_xlim(0, 20)
    #axs[v].set_ylim(bottom=-0.1)

    fig3.set_size_inches(14, 10)
    plt.savefig('../noise/Logistic_Grid_Histogram_of_sigmoidal_slopes_%.0fnpred.pdf' % npred)
    plt.close()


# MEAN(beta_1) across 500
# avg_steepness = np.nanmean(params[:, 0]) # can handle None
# SD(beta_1) across 500
# sd_steepness = np.std(params[:, 0])
# axs[c].text(0, 0.99, 'MEAN(beta_1) = %.1f  SD(beta_1) = %.1f' % (avg_steepness, sd_steepness), horizontalalignment='left', verticalalignment='top')




