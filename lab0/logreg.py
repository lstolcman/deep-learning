import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn.metrics
import data

def logreg_train(X,Y_, param_niter=1000, param_delta=0.1):
    '''
    Arguments
        X:  data, np.array NxD
        Y_: class indices, np.array Nx1
    Return values
        w, b: parameters of binary logistic regression
    '''

    C = max(Y_)+1 # number of classes
    w = np.array([np.random.randn(X.shape[1]) for i in range(C)]) # weights == columns == number of features # CxD
    b = np.array([0.0 for i in range(C)]) # Cx1

    # gradient descent (param_niter iteratons)
    for i in range(param_niter):
        # classification scores size = NxC
        #    classes:
        #    c1 c2 c3 ...
        # n1
        # n2
        # n3 ...
        scores = np.dot(X, w.transpose())+b # NxC
        expscores = np.exp(scores) # NxC
        sumexp = expscores.sum(axis=1) # Nx1

        # calculate probabilities: exp(s_j)/sum(exp(s)) -> sum(exp) - is sum of numbers in area of one class
        probs = (expscores.transpose() / sumexp).transpose() # NxC
        logprobs = np.log(probs) # NxC

        # loss
        loss = sklearn.metrics.log_loss(Y_, probs)# scalar
        # trace
        if i % int(param_niter/10) == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivative of the loss funciton with respect to classification scores
        dL_dscores2 = np.array([(probs - Y_) for probs in probs.transpose()]) # N x C
        dL_dscores = probs   # N x C
        dL_dscores[range(len(X)),Y_] -= 1

        # gradijents with respect to parameters
        # X.shape[0] - number of features
        grad_w = 1/(X.shape[0]) * np.dot(dL_dscores.transpose(), X) # D x C
        grad_b = 1/(X.shape[0]) * dL_dscores.sum(axis=0) # 1 x C

        # modifying the parameters
        w += -param_delta * grad_w
        b += -param_delta * grad_b
    return (w, b)

def logreg_classify(X, w, b):
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

    scores = np.dot(X, w.transpose()) + b    # N x C
    expscores = np.exp(scores)    # N x C

    sumexp = expscores.sum(axis=1)    # N x 1
    return (expscores.transpose() / sumexp).transpose() 


def logreg_decfun(X, W, b):
    def classify(X):
        return logreg_classify(X, W, b).argmax(axis=1)
    return classify


if __name__ == '__main__':
# instantiate the dataset
    np.random.seed(100)
#X, Y_ = sample_gauss_2d(3, 1)
    X=np.array([[1,3],[1,1],[3,2], [3,3]]);Y_=np.array([1,0,2,2])
#X=np.array([[1,1],[2,2]]);Y_=np.array([1,0])

# train the logistic regression model
    w, b = logreg_train(X, Y_, param_niter=1000)

    probs = logreg_classify(X, w, b)
    Y = np.argmax(probs, axis=1)

# graph the decision surface
    decfun = logreg_decfun(X, w, b)
    bbox=(np.min(X-1, axis=0), np.max(X+1, axis=0))


# assign one color to each class
    import matplotlib
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(np.bincount(Y_))))
    graph_surface(decfun, bbox, offset=0.5)
    for i, p in enumerate(X):
        plt.scatter(p[0], p[1], color=colors[Y[i]])

    plt.show()
