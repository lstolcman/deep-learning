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




if __name__ == '__main__':
    np.random.seed(100)
    G = Random2DGaussian()
    X = G.get_sample(100)
    plt.scatter(X[:,0], X[:,1])
    plt.show()


