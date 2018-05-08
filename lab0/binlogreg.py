import numpy as np
import matplotlib.pyplot as plt
import random
import data



import sklearn.metrics

def binlogreg_train(X,Y_, param_niter=1000, param_delta=0.1):
    '''
    Arguments
        X:  data, np.array NxD
        Y_: class indices, np.array Nx1
    Return values
        w, b: parameters of binary logistic regression
    '''
    param_niter += 1 # to print last error
    w = np.random.randn(X.shape[1]) # weights == == columns == number of features
    b = 0.0
    # gradient descent (param_niter iteratons)
    for i in range(param_niter):
        # classification scores
        scores = np.dot(X, w)+b

        #P(Y=c_1|x)=σ(w⊤x+b), where σ(s)=exp(s)/(1+exp(s))
        probs = np.exp(scores)/(1.0+np.exp(scores)) # N x 1
        #probs = 1/(1+np.exp(-scores))

        # loss
        loss = sklearn.metrics.log_loss(Y_, probs)# scalar
        # trace
        if i % int(param_niter/10) == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivative of the loss funciton with respect to classification scores
        #∂L_i/∂s_i = P(c1|xi) − 1{y_i=c1}
        dL_dscores = [probs[i] - Y_[i] for i in range(len(probs))]
        #dL_dscores = probs - Y_ # N x 1

        # gradients with respect to parameters
        N_elements, N_classes = X.shape
        grad_w = 1/N_elements * np.dot(dL_dscores, X) # D x 1
        grad_b = 1/N_elements * np.sum(dL_dscores) # 1 x 1

        # modifying the parameters
        w += -param_delta * grad_w
        b += -param_delta * grad_b
    return (w, b)


def sigma(x):
    return 1/(1+np.exp(x))


def binlogreg_classify(X, w, b):
    '''
      Arguments
          X:    data, np.array NxD
          w, b: logistic regression parameters

      Return values
          probs: a posteriori probabilities for c1, dimensions Nx1
    '''
    #P(Y=c_1|x)=σ(w⊤x+b), where σ(s)=exp(s)/(1+exp(s))
    s = np.dot(X, w)+b
    return sigma(s)


if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X,Y_ = data.sample_gauss_2d(2, 50)

    # train the model
    w,b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w,b)
    Y = np.array([1 if x>0.5 else 0 for x in probs])

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print (accuracy, recall, precision, AP)
