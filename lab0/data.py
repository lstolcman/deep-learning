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


np.random.seed(100)
G = Random2DGaussian()
X = G.get_sample(100)
plt.scatter(X[:,0], X[:,1])
plt.show()


