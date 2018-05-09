import numpy as np
import matplotlib.pyplot as plt
import random


np.random.seed(100)


class Random2DGaussian():
    def __init__(self):
        # [ [minx, maxx], [miny, maxy] ]
        xy = np.array([[0, 10], [0, 10]])

        self.mean = [xy[i][0]+np.random.random_sample()*(xy[i][1]-xy[i][0]) for i in range(2)]
        D = np.diag((np.random.random_sample(2)*(xy[0][1]-xy[0][0], xy[1][1]-xy[1][0]))**2)
        angle = np.random.random_sample()*2*np.pi
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.sigma = np.dot(np.dot(np.transpose(R), D), R)

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.sigma, n)


def sample_gauss_2d(C, N):
    '''
    C - number of classes
    N - number of data points assigned to specific class
    '''
    distributions = []
    for i in range(C):
        distributions.append(Random2DGaussian())

    X = np.vstack([d.get_sample(N) for d in distributions])
    Y_= np.random.randint(0, C, C*N)

    return X,Y_


def eval_perf_binary(Y,Y_):
    tp = sum(np.logical_and(Y==Y_, Y_==True))
    fn = sum(np.logical_and(Y!=Y_, Y_==True))
    tn = sum(np.logical_and(Y==Y_, Y_==False))
    fp = sum(np.logical_and(Y!=Y_, Y_==False))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp+fn + tn+fp)
    return accuracy, recall, precision


def _precision(Y, i):
    class_as_one = Y[i:]
    tp = (class_as_one == 1).sum()
    fp = (class_as_one == 0).sum()
    return tp / (tp + fp)


def eval_AP(Yr):
    Yr = np.array(Yr)
    return np.sum(_precision(Yr, i)*Yr[i] for i in range(len(Yr))) / np.sum(Yr)



import matplotlib as mpl
def graph_data(X, Y_, Y):
    '''
    X  - data (np. array Nx2)
    Y_ - true classes (np.array Nx1)
    Y  - predicted classes (np.array Nx1)
    '''
    
    correct = Y_ == Y
    wrong = Y_ != Y
    
    plt.scatter(X[correct, 0], X[correct, 1], marker='o',
                c=['white' if x else 'grey' for x in Y_[correct]], edgecolors='black')
    plt.scatter(X[wrong, 0], X[wrong, 1], marker='s',
                c=['white' if x else 'grey' for x in Y_[wrong]], edgecolors='black')


def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores






def graph_surface(fun, rect, offset, width=250, height=250):
    '''
      fun    ... the decision function (Nx2)->(Nx1)
      rect   ... he domain in which we plot the data:
                 ([x_min,y_min], [x_max,y_max])
      offset ... the value of the decision function
                 on the border between the classes;
                 we typically have:
                 offset = 0.5 for probabilistic models
                    (e.g. logistic regression)
                 offset = 0 for models which do not squash 
                    classification scores (e.g. SVM)
      width,height ... rezolucija koordinatne mreže
    '''
    (x_min, y_min), (x_max, y_max) = rect
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, width), np.linspace(y_min, y_max, height))
    XX = np.c_[xx.ravel(), yy.ravel()]

    Z = fun(XX).reshape(xx.shape)

    delta = np.abs(Z-offset).max() 
    #plt.pcolormesh(xx, yy, Z, vmin=offset-delta, vmax=offset+delta)
    plt.pcolormesh(xx, yy, Z, vmin=offset-delta, vmax=offset+delta)
    plt.contour(xx, yy, Z, levels=[offset])



def sample_gmm_2d(K, C, N):
    '''
    K - number of distributions
    C - number of classes
    N - number of data points assigned to specific class
    '''
    distributions = []
    classes = []
    for i in range(K):
        distributions.append(data.Random2DGaussian())
        classes.append(np.random.randint(C))
    
    X = np.vstack([d.get_sample(N) for d in distributions])
    Y_ = np.hstack([[c]*N for c in classes])
    return X,Y_


if __name__ == '__main__':
    np.random.seed(100)
    G = Random2DGaussian()
    X = G.get_sample(100)
    plt.scatter(X[:,0], X[:,1])
    plt.show()


