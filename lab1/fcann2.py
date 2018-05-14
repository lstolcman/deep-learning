import numpy as np
import matplotlib.pyplot as plt
import random
import data
import sklearn.metrics

def fcann2_train(X,Y_, param_niter=10000, param_delta=0.05, param_lambda=1e-3, hidden_layer_dim=5):
    '''
    Arguments
        X:  data, np.array NxD
        Y_: class indices, np.array Nx1
    Return values
        w1, b1, w2, b2: parameters
    '''
    num_of_features = X.shape[1]
    num_of_elements = X.shape[0]
    param_delta /= num_of_elements
    C = max(Y_)+1 # number of classes
    # model with 2 layers - one hidden and one exit layer, so w1, b1, w2, b2
    # size of w1,b1 vec. - "5" - hidden_layer_dim - stohastic gradient descent
    w1 = np.array([np.random.randn(num_of_features) for i in range(hidden_layer_dim)])
    b1 = np.array([0.0 for i in range(hidden_layer_dim)])
    w2 = np.array([np.random.randn(hidden_layer_dim) for i in range(C)])
    b2 = np.array([0.0 for i in range(C)]) # Cx1
    
    # stochastic gradient descent (param_niter iteratons)
    for i in range(int(param_niter)):
        # classification scores size = NxC
        #    classes:
        #    c1 c2 c3 ...
        # n1
        # n2
        # n3 ...
        
        #hidden layer
        scores1 = np.dot(X, w1.transpose()) + b1 # NxC
        #ReLU
        h1 = np.maximum(0, scores1) #NxC
        # exit/output layer and softmax calculation
        scores2 = np.dot(h1, w2.transpose())+b2 # NxC
        expscores2 = np.exp(scores2) # NxC
        sumexp2 = expscores2.sum(axis=1) # Nx1
        
        # negative log-likelihood loss
        probs2 = (expscores2.transpose() / sumexp2).transpose() # NxC
        correct_class_prob = probs2[range(len(X)), Y_]
        correct_class_logprobs = -np.log(correct_class_prob) # Nx1
        loss  = correct_class_logprobs.sum()
        
        # trace
        if i % int(param_niter/10) == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivative of the loss funciton with respect to classification scores

        dscores2 = probs2 # NxC
        # @lab0:
        # Using Iversonovim brackets to denote the influence of the corresponding
        # input category, we can then define the partial derivative of the loss function
        # with respect to the classification score
        dscores2[range(num_of_elements),Y_] -= 1
        
        dw2 = np.dot(dscores2.transpose(), h1) # CxH
        db2 = dscores2.sum(axis=0) # Cx1

        dh1 = np.dot(dscores2, w2)  # NxH

        dscores1 = dh1  # NxH
        dscores1[scores1 <= 0] = 0

        dw1 = np.dot(dscores1.transpose(), X) # HxD
        db1 = dscores1.sum(axis=0)  # Hx1

        # update
        w1 += -param_delta * dw1
        b1 += -param_delta * db1
        w2 += -param_delta * dw2
        b2 += -param_delta * db2

    return (w1, b1, w2, b2)

def fcann2_classify(X, w1, b1, w2, b2):
    '''
    Parameters
    ----------
    X : 2-D array_like, of shape (N, 2)
        data
    w : 2-D array_like, of shape (C, 2)
        array of weights for each feature
    b: 1-D array like of length C
        bias parameter

    Returns
    -------
    probs : 2-D array_like, of shape (N, C)
        probability of classes for each sample in X
    '''
    scores1 = np.dot(X, w1.transpose()) + b1 # NxH
    #ReLU
    h1 = np.maximum(0, scores1) # NxH
    scores2 = np.dot(h1, w2.transpose()) + b2 # NxC

    #softmax
    expscores = np.exp(scores2) # NxC
    sumexp = expscores.sum(axis=1) # Nx1

    return (expscores.transpose() / sumexp).transpose() 


if __name__ == '__main__':
    np.random.seed(32)
    X, Y = sample_gmm_2d(6, 2, 10)
    ##X=np.array([[1,3],[1,1],[3,2], [3,3]]);Y=np.array([1,0,2,2])
    w1, b1, w2, b2 = fcann2_train(X,Y)

    Y_ = np.empty((0,0), dtype=np.int64)
    # get class by doing argmax on vector (highest value = predicted class)
    ##for v in fcann2_classify(X, w1, b1, w2, b2):
    ##    Y_ = np.append((Y_), np.argmax(v))
    Y_= np.argmax(fcann2_classify(X, w1, b1, w2, b2), axis=1) # axis=1 - horizontally

    accuracy, conf_mat, prec_recall = data.eval_perf_multi(Y,Y_)
    print('accuracy:', accuracy)

    plt.scatter(X[:,0], X[:,1], c=Y)
    plt.show()
