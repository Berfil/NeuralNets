import numpy as np
from scipy.stats import mode 
import math



def euclidian_distance(row_1, row_2):
        
        distance = np.sqrt(np.sum((row_1-row_2)**2))
        
        return(distance)
    
    
def kNN(X, k, XTrain, LTrain):
    """ KNN
    Your implementation of the kNN algorithm

    Inputs:
            X      - Samples to be classified (matrix)
            k      - Number of neighbors (scalar)
            XTrain - Training samples (matrix)
            LTrain - Correct labels of each sample (vector)

    Output:
            LPred  - Predicted labels for each sample (vector)
    """
    labels_return = np.array([])
    
    for i in range(X.shape[0]):
        current_point_dist = np.array([])
        
        for j in range(len(XTrain)):
            distance = euclidian_distance(np.array(XTrain[j,:]) , X[i,:])
            current_point_dist= np.append(current_point_dist , distance)

        current_point_dist = np.array(current_point_dist)
        
        distance_array = np.argsort(current_point_dist)[:k]
        label = LTrain[distance_array]
        lab = mode(label)[0][0]
       

        
        labels_return= np.append(labels_return, lab)
        labels_return = np.array(labels_return , dtype = np.int64)
   
    return labels_return

def runSingleLayer(X, W):
    """ RUNSINGLELAYER
    Performs one forward pass of the single layer network, i.e
    it takes the input data and calculates the output for each sample.

    Inputs:
            X - Samples to be classified (matrix)
            W - Weights of the neurons (matrix)

    Output:
            Y - Output for each sample and class (matrix)
            L - The resulting label of each sample (vector)
    """
    

    #W = np.random.rand(len(X[1]),len(D[1]))
    Y = np.matmul(X , W )
    

    # Calculate labels
    L = np.argmax(Y, axis=1) + 1

    return Y, L


def single_gradient(XTrain , Wout , DTrain):
    
    DW = np.zeros(Wout.shape)
    for p in range(Wout.shape[0]):
        for q in range(Wout.shape[1]):
            A = np.dot(XTrain , Wout[:,q])
            DW[p,q] = (2/len(XTrain)) *  np.dot(A - DTrain[:,q] , XTrain[:,p].T ) 
    return DW



def trainSingleLayer(XTrain, DTrain, XTest, DTest, W0, numIterations, learningRate):
    """ TRAINSINGLELAYER
    Trains the single-layer network (Learning)

    Inputs:
            X* - Training/test samples (matrix)
            D* - Training/test desired output of net (matrix)
            W0 - Initial weights of the neurons (matrix)
            numIterations - Number of learning steps (scalar)
            learningRate  - The learning rate (scalar)
    Output:
            Wout - Weights after training (matrix)
            ErrTrain - The training error for each iteration (vector)
            ErrTest  - The test error for each iteration (vector)
    """

    # Initialize variables
    ErrTrain = np.zeros(numIterations+1)
    ErrTest  = np.zeros(numIterations+1)
    NTrain = XTrain.shape[0]
    NTest  = XTest.shape[0]
    Wout = W0

    # Calculate initial error
    YTrain , LTrain = runSingleLayer(XTrain, Wout)
    YTest , LTest = runSingleLayer(XTest , Wout)
    ErrTrain[0] = ((YTrain - DTrain)**2).sum() / NTrain
    ErrTest[0]  = ((YTest  - DTest )**2).sum() / NTest

    for n in range(numIterations):
        # Add your own code here
        grad_w = single_gradient(XTrain , Wout , DTrain)

        # Take a learning step
        Wout = Wout - learningRate * grad_w

        # Evaluate errors
        YTrain, LTrain = runSingleLayer(XTrain, Wout)
        YTest, LTest = runSingleLayer(XTest , Wout)
        ErrTrain[n+1] = ((YTrain - DTrain) ** 2).sum() / NTrain
        ErrTest[n+1]  = ((YTest  - DTest ) ** 2).sum() / NTest

    return Wout, ErrTrain, ErrTest


def runMultiLayer(X, W, V):
    """ RUNMULTILAYER
    Calculates output and labels of the net

    Inputs:
            X - Data samples to be classified (matrix)
            W - Weights of the hidden neurons (matrix)
            V - Weights of the output neurons (matrix)

    Output:
            Y - Output for each sample and class (matrix)
            L - The resulting label of each sample (vector)
            H - Activation of hidden neurons (vector)
    """
    
    b = np.matmul(X,W)
   
    c = np.tanh(b)
 
    d = np.matmul(c , V)
    

    
    H = c  # Calculate the activation of the hidden neurons (use hyperbolic tangent)
    Y = d  # Calculate the weighted sum of the hidden neurons

    # Calculate labels
    L = Y.argmax(axis=1) + 1

    return Y, L, H


def grad_v(NTrain , HTrain , YTrain , DTrain):
    return 2/NTrain * np.matmul(HTrain.transpose(), YTrain-DTrain)
        
def grad_w(NTrain , XTrain, YTrain , DTrain, Vout, HTrain):
    return 2/NTrain * np.matmul(XTrain.transpose(), (np.multiply(np.matmul(YTrain-DTrain, Vout.transpose()), (1-HTrain**2))))


def trainMultiLayer(XTrain, DTrain, XTest, DTest, W0, V0, numIterations, learningRate):
    """ TRAINMULTILAYER
    Trains the multi-layer network (Learning)

    Inputs:
            X* - Training/test samples (matrix)
            D* - Training/test desired output of net (matrix)
            V0 - Initial weights of the output neurons (matrix)
            W0 - Initial weights of the hidden neurons (matrix)
            numIterations - Number of learning steps (scalar)
            learningRate  - The learning rate (scalar)

    Output:
            Wout - Weights after training (matrix)
            Vout - Weights after training (matrix)
            ErrTrain - The training error for each iteration (vector)
            ErrTest  - The test error for each iteration (vector)
    """

    # Initialize variables
    ErrTrain = np.zeros(numIterations+1)
    ErrTest  = np.zeros(numIterations+1)
    NTrain = XTrain.shape[0]
    NTest  = XTest.shape[0]
    NClasses = DTrain.shape[1]
    Wout = W0
    Vout = V0

    # Calculate initial error
    # YTrain = runMultiLayer(XTrain, W0, V0)
    YTrain, LTrain , HTrain = runMultiLayer(XTrain, Wout, Vout)
    YTest, LTest , HTest  = runMultiLayer(XTest , W0, V0)
    ErrTrain[0] = ((YTrain - DTrain)**2).sum() / (NTrain * NClasses)
    ErrTest[0]  = ((YTest  - DTest )**2).sum() / (NTest * NClasses)

    for n in range(numIterations):

        if not n % 1000:
            print(f'n : {n:d}')

        # Add your own code here
        gradient_v = grad_v(NTrain , HTrain , YTrain , DTrain) # Gradient for the output layer
        gradient_w = grad_w(NTrain , XTrain, YTrain , DTrain, Vout, HTrain) # And the input layer

        # Take a learning step
        Vout = Vout - learningRate * gradient_v
        Wout = Wout - learningRate * gradient_w

        # Evaluate errors
        YTrain, LTrain, HTrain = runMultiLayer(XTrain, Wout, Vout)
        YTest, LTest, HTest  = runMultiLayer(XTest , Wout, Vout)
        ErrTrain[1+n] = ((YTrain - DTrain)**2).sum() / (NTrain * NClasses)
        ErrTest[1+n]  = ((YTest  - DTest )**2).sum() / (NTest * NClasses)

    return Wout, Vout, ErrTrain, ErrTest