import numpy as np
import matplotlib.pyplot as plt
import random
import data
import tensorflow as tf


class TFLogreg:
    def __init__(self, D, C, param_delta=0.5, param_lambda=1e-3):
        """Arguments:
           - D: dimensions of each datapoint 
           - C: number of classes
           - param_delta: training step
        """

        # declare graph nodes for the data and parameters:
        # self.X, self.Yoh_, self.W, self.b
        self.X = tf.placeholder(tf.float64, [None, D])
        self.Yoh_ = tf.placeholder(tf.float64, [None,C])
        self.W = tf.Variable(np.array([np.random.randn(D) for i in range(C)])) # weights == columns == number of features # CxD
        self.b = tf.Variable(np.array([0.0 for i in range(C)]))

        # formulate the model: calculate self.probs
        #   use tf.matmul, tf.nn.softmax
        scores = tf.matmul(self.X, self.W, transpose_b=True) + self.b
        self.probs = tf.nn.softmax(scores)

        # formulate the loss: self.loss
        #   use tf.log, tf.reduce_sum, tf.reduce_mean
        # L(W,b|D)=∑_i−logP(Y=y_i|x_i)
        minuslog = -tf.log(self.probs)
        sumlog = tf.reduce_sum(self.Yoh_ * minuslog, axis=1)
        self.loss = tf.reduce_mean(sumlog)
        # add l2 regularization
        '''
        @ Ian Goodfellow, Deep Learning, 5.2.2:
        λ is a value chosen ahead of time that controls the strength of our preference
        for smaller weights. When λ = 0, we impose no preference, and larger λ forces the
        weights to become smaller. Minimizing J ( w ) results in a choice of weights that
        make a tradeoff between fitting the training data and being small.
        '''
        self.loss += param_lambda * tf.nn.l2_loss(self.W)

        # formulate the training operation: self.train_step
        #   use tf.train.GradientDescentOptimizer,
        #       tf.train.GradientDescentOptimizer.minimize
        opt = tf.train.GradientDescentOptimizer(0.1) # defining step
        self.train_step = opt.minimize(self.loss) # minimizing loss

        # instantiate the execution context: self.session
        #   use tf.Session
        self.session = tf.Session()


    def train(self, X, Yoh_, param_niter):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """
        # parameter intiailization
        #   use tf.global_variables_initializer !! deprecated
        self.session.run(tf.global_variables_initializer())

        data_in = {self.X: X, self.Yoh_: Yoh_}
        # optimization loop
        #   use tf.Session.run
        ## tf.Session = self.session, so self.session.run()
        for i in range(param_niter):
            loss_val = self.session.run(self.loss, feed_dict=data_in)
            self.session.run(self.train_step, feed_dict=data_in)
            if i%100==0:
                print('{} loss: {}'.format(i, loss_val))
        print('{} loss: {}'.format(i, loss_val))


    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        #   use tf.Session.run
        ## tf.Session = self.session, so self.session.run()
        return self.session.run(self.probs, feed_dict={self.X: X})


if __name__ == "__main__":
    # initialize the random number generator
    np.random.seed(100)
    tf.set_random_seed(100)
    tf.reset_default_graph() 

    # instantiate the data X and the labels Yoh_
    X = np.array([[1,3],[1,1],[3,2],[3,3]])
    Y_ = np.array([1,0,2,2])
    '''
    Yoh c1 c2 c3
     x1 0  1  0
     x2 1  0  0
     x3 0  0  1
     x4 0  0  1
    '''
    Yoh_ = np.zeros((len(X), len(np.bincount(Y_))))
    Yoh_[range(len(Y_)), Y_] = 1
    ## Yoh_s = tf.one_hot(D,C)

    '''
    N_classes = 3
    N_examples = 2 # examples from each class, so 3*10=30 in total
    X, Y_ = data.sample_gauss_2d(N_classes, N_examples)
    Yoh_ = np.zeros((N_examples*N_classes, N_classes))
    Yoh_[range(len(Y_)), Y_] = 1
    Yoh_ = tf.one_hot(Y_, N_classes)
    '''

    # build the graph:
    tflr = TFLogreg(X.shape[1], Yoh_.shape[1], 0.5)

    # perform the training with given hyper-parameters:
    tflr.train(X, Yoh_, 1000)

    # predict probabilities of the data points
    probs = tflr.eval(X)

    # print performance (per-class precision and recall)

    # draw results, decision surface
    decfun = lambda x: tflr.eval(x).argmax(axis=1)
    bbox=(np.min(X-1, axis=0), np.max(X+1, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    '''
    ## assign one color to each class
    import matplotlib
    c = matplotlib.cm.rainbow(np.linspace(0, 1, len(np.bincount(Y_))))
    for i, p in enumerate(X):
        plt.scatter(p[0], p[1], color=c[Y_[i]])
    '''
    #Y = [np.argmax(p) for p in probs]
    Y = np.argmax(probs, axis=1)

    data.graph_data(X, Y_, Y)
    plt.show()

