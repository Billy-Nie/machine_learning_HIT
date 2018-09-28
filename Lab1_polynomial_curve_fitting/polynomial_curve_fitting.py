import numpy as np
from numpy.linalg import inv, det
import math
import matplotlib.pyplot as plt
import time

def data_generator(n, mu, sigma, start_value):
    '''
    Generate n data drawn from sin(x) (0 < x < 1) with noise drawn from
    Gaussian distribution added to each of the data
    :param n: the size of the data set
    :param mu: mu for the Gaussian distribution
    :param sigma: sigma for the Gaussian distribution
    :return: a n dimensional list of x-coordinate, y a n dimensional list of data generated
    '''

    x = [start_value + x / n for x in range(0, n)]
    y = [math.sin(2*math.pi*i) for i in x]
    return np.array(x), y + np.random.normal(mu, sigma, n)

def polynomial_result(fitter,x):
    '''
    calculate the corresponding polynomial result of x
    :param W: the constant term W
    :param x: the x
    :param m: the order of polynomial term
    :return: the corresponding polynomial result
    '''
    result = 0
    for i in range(0, fitter.get_m() + 1):
        result += fitter.get_W()[i] * (x ** i)
    return result

def rms(W,x,y,m,n):
    '''
    calculate the root mean square
    :return RMS
    '''
    X = []
    for i in x:
        X.append([i ** j for j in range(0, m + 1)])
    X = np.array(X)
    y = y.reshape(-1,1)
    error = (X.dot(W) - y).transpose().dot(X.dot(W) - y)
    rms = math.sqrt(error /n)
    return rms

def plot_rms_trend(x, rms_train_L, rms_test_L, fit_method):
    plt.plot(x, rms_train_L, label = "training data")
    plt.plot(x, rms_test_L, label="test data")
    plt.scatter(x,rms_train_L)
    plt.scatter(x, rms_test_L)
    plt.xlabel("learning rate")
    plt.ylabel("$E_{RMS}$")
    plt.legend()
    plt.title("$E_{RMS}$ trend with different learning rate")
    plt.savefig("Latex-Template/src/ML/figures/" + str(fit_method) + ".jpg")
    plt.show()

class PolynomialFitter:
    def __init__(self, m, x, y):
        '''
        :param m: the order of polynomial term
        :param x: the x-cooridinate of training data
        :param y: the data set generated from data_generaton
        '''
        self.__m = m
        self.__x = x
        self.__y = y
        self.__y = y.reshape(-1,1)
        self.__W = None
        self.__RMS = None # the root-mean-square
        X = []
        for i in self.__x:
            X.append([i ** j for j in range(0, self.__m + 1)])
        X = np.array(X)  # X is the n * (m + 1) dimensional matrix
        self.__X = X
        self.__plottable = True

    def get_m(self):
        return self.__m

    def get_X(self):
        return self.__X

    def get_W(self):
        return self.__W

    def get_RWS(self):
        return  self.__RMS

    def get_y(self):
        return self.__y

    def root_mean_square(self):
        '''
        calculate the root mean square
        set self.RMS
        '''
        error = (self.__X.dot(self.__W) - self.__y).transpose().dot(self.__X.dot(self.__W) - self.__y)
        rms = math.sqrt(error / len(self.__x))
        self.__RMS = rms
        return rms

    def plot_training(self, fit_method):
        '''
        plot how well the polynomial curve fitting is with contrast to
        the real y = sin(2πx)
        '''

        if self.__plottable:
            a = [i / 100 for i in range(0, 100)]
            b = [math.sin(math.pi * 2 * i) for i in a]
            c = [polynomial_result(fitter, i) for i in a]
            y = [math.sin(2 * math.pi * i) for i in x]
            plt.plot(a, b, label="sin($2\pi x$)")
            plt.plot(a, c, label="M = " + str(self.__m))
            plt.scatter(self.__x, y, label="sin($2\pi x$)")
            plt.scatter(self.__x, self.__y, label="Data set")
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.title(fit_method)
            plt.savefig("Latex-Template/src/ML/figures/" + str(fit_method) + ".jpg")
            plt.show()

        else:
            raise ValueError("Currently not plottable may because the numerical solution is " + \
                             "not supported try gradient decent")


    def fit_without_regular_term(self):
        '''
        calculate the constant term w using numerical solution
        without the regulation term
        :set self.__W to a (m + 1) * 1 ndarray
        '''
        if det(self.__X.transpose().dot(self.__X)) != 0.0:
            W = inv(self.__X.transpose().dot(self.__X)).dot(self.__X.transpose()).dot(self.__y)
            self.__W = W
        else:
            raise ValueError("numerical solution not supported please use gradient decent instead")

    def fit_with_regular_term(self, lam):
        '''
        calculate the constant term w using numerical solution
        with regulation term
        :param lam: lambda of the regulation term often called hyper-parameter
        :set: self.__W to a (m + 1) * 1 ndarray
        '''
        if det(self.__X.transpose().dot(self.__X)) != 0.0:
            W = inv(self.__X.transpose().dot(self.__X) + lam * np.eye((self.__m+1),(self.__m+1))).dot(self.__X.transpose()).dot(self.__y)
            self.__W = W
        else:
            raise ValueError("numerical solution not supported please use gradient decent instead")

    def fit_with_gradient_descent(self, lr, lam):
        '''
        fitting the curve using gradient descent
        :param lr: learning rate of gradient descent
        :param lam: the lambda of regulation term often called hyper-parameter
        :set: self.__W to a (m + 1) * 1 ndarray
        '''
        #init value for __W
        start = time.time()
        W = []
        for i in range(self.__m + 1):
            W.append(0)
        W = np.array(W).reshape(-1, 1)
        self.__W = W
        iteration_thresh = 1e10
        iter = 0
        thresh = 1e-7
        while True:
            iter += 1
            term_vector = self.__X.dot(W) - self.__y
            error_before = self.root_mean_square()
            W = W - lr / len(self.__x) *  self.__X.transpose().dot(term_vector) + lam * W
            self.__W = W
            error_after = self.root_mean_square()
            if np.abs(error_before - error_after) <= thresh or iter == iteration_thresh:
                break
            print("iter:{0}\terror:{1}\n".format(iter, error_after))
        end = time.time()
        print("Run time : {0}, with RMS : {1}".format(round((end-start), 2), round(error_after,2)))
        return round((end-start), 2)

    def fit_with_stochastic_gradient_descent(self, lr, lam):
        '''
        fitting the curve using stochastic gradient descent
        which is a faster way to calculate
        :param lr: learning rate of stochastic gradient descent
        :param lam: the lambda of regulation term often called hyper-parameter
        :set: self.__W to a (m + 1) * 1 ndarray
        '''

        start = time.time()
        W = []
        for i in range(self.__m + 1):
            W.append(0)
        W = np.array(W).reshape(-1, 1)
        self.__W = W
        iteration_thresh = 400000
        iter = 0
        i = 0
        thresh = 1e-9
        while True:
            diff = self.__y[i] - self.__X[i].reshape(1, -1).dot(self.__W)
            error_before = self.root_mean_square()
            W = W + lr * diff * self.__X[i].reshape(-1, 1) + lam * W
            self.__W = W
            error_after = self.root_mean_square()
            if np.abs(error_before - error_after) < thresh or iter == iteration_thresh:
                break
            i = (i + 1) % len(self.__x)
            iter += 1
            print("iter:{0}\terror:{1}\n".format(iter, error_after))

        end = time.time()
        print("Run time : {0}, with RMS : {1}".format(round((end-start), 2), round(error_after,2)))

    def fit_with_conjugate_gradient_method(self, lam):
        '''
        fitting the curve with conjugate gradient method
        :param lam: the lambda for penalty
        :set: self.__W to a (m+1) ndarray
        '''

        W = [0 for i in range(self.__m + 1)]
        W = np.array(W).reshape(-1, 1) # reshape to column vector
        self.__W = W # set initial value for W
        y = self.__y.reshape(-1, 1)
        y = self.__X.transpose().dot(y)
        X = self.__X.transpose().dot(self.__X) + lam * np.eye(self.__X.shape[1], self.__X.shape[1])
        r = y - X.dot(self.__W)
        p = r
        rsold = r.transpose().dot(r)
        for i in range(len(self.__x)):
            Xp = X.dot(p)
            alpha = rsold / (p.transpose().dot(Xp))
            self.__W = self.__W  + alpha * p
            r = r - alpha * Xp
            rsnew = r.transpose().dot(r)
            if rsnew < 1e-6:
                break
            else:
                p = r + rsnew/rsold * p
                rsold = rsnew

if __name__ == "__main__":
    x,t = data_generator(10, 0, 0.1, 0)
    x2,t2 = data_generator(10,0,0.1,0.05)
    fitter = PolynomialFitter(9, x, t)


    rms_train_L = []
    rms_test_L = []
    #for i in range(3,10):
    #    fitter = PolynomialFitter(i,x,t)
    #    fitter.fit_without_regular_term()
    #    fitter.plot_training("loss_no_reg_" + str(i))
    #    rms_train = fitter.root_mean_square()
    #    rms_train_L.append(rms_train)
    #    rms_test = rms(fitter.get_W(), x2, t2, fitter.get_m(), len(x2))
    #    rms_test_L.append(rms_test)

    #for i in range(10,110,10):
    #    x, t = data_generator(i, 0, 0.1, 0)
    #    x2, t2 = data_generator(i, 0, 0.1, 0.05)
    #    fitter = PolynomialFitter(9, x, t)
    #    fitter.fit_without_regular_term()
    #    fitter.plot_training("loss_no_reg_n_" + str(i))
    #    rms_train = fitter.root_mean_square()
    #    rms_train_L.append(rms_train)
    #    rms_test = rms(fitter.get_W(), x2, t2, fitter.get_m(), len(x2))
    #    rms_test_L.append(rms_test)

    #plot_x = [i for i in range(3,10)]
    #plot_rms_trend(plot_x, rms_train_L, rms_test_L, "loss_no_reg_trend")

    #fitter.fit_without_regular_term()
    #fitter.plot_training("numerical solution without regulation term")
    #print("训练集上的RMS:" + str(fitter.root_mean_square()))
    #x2,t2 = data_generator(10,0,0.1,0.05)
    #rms2 = rms(fitter.get_W(),x2,t2,fitter.get_m(),len(x2))
    #print("测试集上的RMS:" + str(rms2))

    #fitter.fit_with_regular_term(np.e ** -0)
    #fitter.plot_training("a")
    #print("测试集上的RMS:" + str(fitter.root_mean_square()))
    #rms2 = rms(fitter.get_W(), x2, t2, fitter.get_m(), len(x2))
    #print("测试集上的RMS:" + str(rms2))

    #for i in range(-10,0,2):
    #    fitter.fit_with_regular_term(np.e **(i))
    #    fitter.plot_training("loss_reg_" + str(i))
    #    rms_train = fitter.root_mean_square()
    #    rms_train_L.append(rms_train)
    #    rms_test = rms(fitter.get_W(), x2, t2, fitter.get_m(), len(x2))
    #    rms_test_L.append(rms_test)
    #plot_x = [i for i in range(-10, 0, 2)]
    #plot_rms_trend(plot_x, rms_train_L, rms_test_L, "loss_reg_trend")


    #fitter.fit_with_gradient_descent(0.02, np.e ** -18)
    #fitter.plot_training("gradient_descent")
    #print("测试集上的RMS:" + str(fitter.root_mean_square()))
    #rms2 = rms(fitter.get_W(), x2, t2, fitter.get_m(), len(x2))
    #print("测试集上的RMS:" + str(rms2))

    #fitter.fit_with_stochastic_gradient_descent(0.1, np.e ** -20)
    #fitter.plot_training("stochastic_gradient_descent")
    #print("测试集上的RMS:" + str(fitter.root_mean_square()))
    #rms2 = rms(fitter.get_W(), x2, t2, fitter.get_m(), len(x2))
    #print("测试集上的RMS:" + str(rms2))


    #lr_L = [i / 100 for i in range(2,12,2)]
    #time_L = []
    #for i in lr_L:
    #    run_time = fitter.fit_with_gradient_descent(i, np.e ** -18)
    #    time_L.append(run_time)
    #    fitter.plot_training("gradient_descent_" + str(i * 100))
    #    rms_train = fitter.root_mean_square()
    #    rms_train_L.append(rms_train)
    #    rms_test = rms(fitter.get_W(), x2, t2, fitter.get_m(), len(x2))
    #    rms_test_L.append(rms_test)

    #plot_rms_trend(lr_L, rms_train_L, rms_test_L, "gradient_descent_learning_rate")
    #plt.plot(lr_L, time_L)
    #plt.xlabel('learning rate')
    #plt.ylabel("time/s")
    #plt.title("run time with change to learning rate")
    #plt.savefig("Latex-Template/src/ML/figures/gradient_descent_time.jpg")
    #plt.show()

    fitter.fit_with_conjugate_gradient_method(np.e ** -13)
    fitter.plot_training("conjugate_gradient_method")
    print(fitter.root_mean_square())
    rms2 = rms(fitter.get_W(), x2, t2, fitter.get_m(), len(x2))
    print("测试集上的RMS:" + str(rms2))





