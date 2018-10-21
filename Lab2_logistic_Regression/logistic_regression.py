import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def data_generator(n, mu, sigma, seed = 0):
    '''
    generate a set of data according to the Gaussian Distribution
    :param n: the size of data set
    :param mu: mu in the Gaussian Distribution
    :param sigma: sigma in the Gaussian Distribution
    :return:  according to the Gaussian Distribution
    '''

    np.random.seed(seed)
    x = np.random.normal(mu, sigma, n)
    y = np.random.normal(mu, sigma, n)
    data = [[x[i], y[i]] for i in range(n)]
    #print(data)
    return [x, y], data

def plot_data(data_l, w, title, n = 2, lr = False):
    '''
    plot the dataset using scatter
    :param n: the number of dataset
    :param data_l: a list of data
    :param w: the weight calculated in logistic regression
    :param lr: boolean whether to plot the decision boundary or not
    :return: a plot generated by the scatter
    '''

    for i in range(n):
        plt.scatter(data_l[i][0], data_l[i][1], label="dataset" + str(i))
        if lr:
            x = [i / 10 for i in range(-10, 30)]
            y = [- (w[0][0] + w[1][0] * i) / w[2][0] for i in x]
            plt.plot(x, y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.show()

def read_data():
    '''
    read the data downloaded from uci
    :return: the real data for later analyzation
    '''

    file = open("banknote authentication.txt", "r")
    lines = file.readlines()
    data = []
    label = []
    for line in lines:
        line_data = []
        line = line.strip()
        line_split = line.split(',')
        label.append(float(line_split[-1]))
        for i in range(len(line_split) - 1):
            line_data.append(float(line_split[i]))
        data.append(line_data)

    return data, label

class LR:
    def __init__(self, data_set, label):
        assert len(data_set) == len(label)
        self.__m = len(data_set)
        self.__n = len(data_set[0]) + 1
        self.__data = [[1] + (data_set[i]) for i in range(self.__m)]
        self.__data = np.array(self.__data)
        self.__label = label
        self.__label = np.array(label).reshape(-1, 1) # reshape to a column vector
        #self.__w = np.random.normal(0, 1, self.__n).reshape(-1, 1) # assume the prior as a Gaussian Distribution
        self.__w = np.array([0 for i in range(self.__n)]).reshape(-1,1)
        self.__error = None # the error of the model

    def sigmoid(self, z):
        '''
        the sigmoid function
        see more at https://en.wikipedia.org/wiki/Sigmoid_function
        '''
        return 1 / (1 + np.e ** -z)

    def cost(self, h,label):
        cost = 0
        for i in range(len(label)):
            if label[i] == 1:
                cost += (np.log(h[i]))
            else:
                cost += (np.log(1-h[i]))
        return cost

    def plot_training(self, iteration_l, FPV_l, ACC_l, cost_l, newton = True):

        if newton:
            plt.scatter(iteration_l, cost_l)
            plt.plot(iteration_l, cost_l, label = "cost")
            plt.xlabel("iterations")
            plt.legend()
            plt.title("Cost as function of iterations")
            plt.ylabel("cost")
            plt.show()

            plt.scatter(iteration_l, FPV_l)
            plt.plot(iteration_l, FPV_l, label = "FPV")
            plt.scatter(iteration_l, ACC_l)
            plt.plot(iteration_l, ACC_l, label = "ACC")
            #plt.scatter(iteration_l, F1_l)
            #plt.plot(iteration_l, F1_l, label = "F1 score")
            plt.legend()
            plt.xlabel("iterations")
            plt.ylabel("y")
            plt.title("model evaluation factors")
            plt.show()
        else:
            plt.plot(iteration_l, cost_l, label="cost")
            plt.xlabel("iterations")
            plt.legend()
            plt.title("Cost as function of iterations")
            plt.ylabel("cost")
            plt.show()

            plt.plot(iteration_l, FPV_l, label="FPV")
            plt.plot(iteration_l, ACC_l, label="ACC")
            plt.legend()
            plt.xlabel("iterations")
            plt.ylabel("y")
            plt.title("model evaluation factors")
            plt.show()


    def get_w(self):
        return self.__w

    def get_data(self):
        return self.__data

    def gradAscent(self):
        '''
        learning the model using gradient ascent
        :return: the weight
        '''

        alpha = 0.01#learning rate for self generated data
        #alpha = 0.001 # learning rate for uci data
        max_iter = 1000 #最大循环数
        cost_l = []
        iteration_plot = []
        FPV_l = []
        ACC_l = []
        F1_l = []
        for i in range(max_iter):
            h = self.sigmoid(self.__data.dot(self.__w)) # m * 1
            error = self.cost(h, self.__label)
            print("iterate :" + str(i + 1))
            print("cost:" + str(error))
            self.__w = self.__w + alpha * self.__data.transpose().dot(self.__label - h)
            if i % int(max_iter/50) == 0 and i >= int(max_iter / 50):
                FPV,ACC,F1 = self.extract_evalutation(self.__data)
                cost_l.append(error)
                FPV_l.append(FPV)
                ACC_l.append(ACC)
                F1_l.append(F1)
                iteration_plot.append(i + 1)
        self.plot_training(iteration_plot, FPV_l, ACC_l, cost_l, False)
        h = self.sigmoid(self.__data.dot(self.__w))
        print("result error" + str(float(self.cost(h, self.__label)))+"\n")
        FPV, ACC, F1 = self.extract_evalutation(self.__data)
        print("fall out: " + str(FPV))
        print("accuracy:" + str(ACC))
        print("F1 score:" + str(F1))

    def gradAscentReg(self):
        '''
        learning the model using gradient ascent with regulation term
        :return: the weight
        '''
        alpha = 0.01 # learning rate for self generated data
        #alpha = 0.001 # learning rate for uci data
        eta = 0.01
        max_iter = 600  # 最大循环数
        cost_l = []
        iteration_plot = []
        FPV_l = []
        ACC_l = []
        for i in range(max_iter):
            h = self.sigmoid(self.__data.dot(self.__w))  # m * 1
            error = self.cost(h, self.__label)
            print("iterate :" + str(i + 1))
            print("error:" + str(float(error)))
            self.__w = self.__w + alpha * eta * self.__w + alpha * self.__data.transpose().dot(self.__label - h)
            if i % int(max_iter/50) == 0 and i >= int(max_iter / 50):
                FPV,ACC,F1 = self.extract_evalutation(self.__data)
                cost_l.append(error)
                FPV_l.append(FPV)
                ACC_l.append(ACC)
                iteration_plot.append(i + 1)
        self.plot_training(iteration_plot, FPV_l, ACC_l, cost_l, False)
        h = self.sigmoid(self.__data.dot(self.__w))
        print("result error" + str(float(self.cost(h, self.__label))) + "\n")
        FPV, ACC, F1 = self.extract_evalutation(self.__data)
        print("fall out: " + str(FPV))
        print("accuracy:" + str(ACC))
        print("F1 score:" + str(F1))

    def newton(self):
        max_iter = 20 #最大循环数
        cost_l = []
        iteration_plot = []
        FPV_l = []
        ACC_l = []
        for i in range(max_iter):
            h = self.sigmoid(self.__data.dot(self.__w)) # m + 1
            A = np.diag([float(h[i] * (1 - h[i])) for i in range(len(h))])
            H = self.__data.transpose().dot(A).dot(self.__data)
            error = self.__label - h
            U = self.__data.transpose().dot(error)
            self.__w = self.__w + inv(H).dot(U)
            error = self.cost(h, self.__label)
            print("iterate :" + str(i + 1))
            print("error:" + str(float(error)))
            FPV,ACC,F1 = self.extract_evalutation(self.__data)
            cost_l.append(error)
            FPV_l.append(FPV)
            ACC_l.append(ACC)
            iteration_plot.append(i + 1)
        self.plot_training(iteration_plot, FPV_l, ACC_l, cost_l, True)
        h = self.sigmoid(self.__data.dot(self.__w))
        print("result error" + str(float(self.cost(h, self.__label))) + "\n")
        FPV, ACC, F1 = self.extract_evalutation(self.__data)
        print("fall out: " + str(FPV))
        print("accuracy:" + str(ACC))
        print("F1 score:" + str(F1))

    def newtonReg(self):
        max_iter = 10  # 最大循环数
        eta = 0.0001
        cost_l = []
        iteration_plot = []
        FPV_l = []
        ACC_l = []
        for i in range(max_iter):
            h = self.sigmoid(self.__data.dot(self.__w))  # m + 1
            A = np.diag([float(h[i] * (1 - h[i])) for i in range(len(h))])
            H = self.__data.transpose().dot(A).dot(self.__data)
            error = self.__label - h
            U = self.__data.transpose().dot(error)
            self.__w = self.__w + inv(H).dot(U + eta * self.__w)
            error = self.cost(h, self.__label)
            print("iterate :" + str(i + 1))
            print("error:" + str(float(error)))
            FPV,ACC,F1 = self.extract_evalutation(self.__data)
            cost_l.append(error)
            FPV_l.append(FPV)
            ACC_l.append(ACC)
            iteration_plot.append(i + 1)
        self.plot_training(iteration_plot, FPV_l, ACC_l, cost_l, True)
        h = self.sigmoid(self.__data.dot(self.__w))
        print("result error" + str(float(self.cost(h, self.__label))) + "\n")
        FPV, ACC, F1 = self.extract_evalutation(self.__data)
        print("fall out: " + str(FPV))
        print("accuracy:" + str(ACC))
        print("F1 score:" + str(F1))

    def extract_evalutation(self, data):
        '''
        extract the fall out(false positive rate), accuracy, F1 score for the model
        :return: FPV, ACC, F1 score
        '''
        h = self.sigmoid(data.dot(self.__w))
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        label_predict = []
        for i in h:
            if i < 0.5:
                label_predict.append(0)
            else:
                label_predict.append(1)

        for i in range(self.__m):
            if self.__label[i] == label_predict[i] and self.__label[i] == 1:
                TP += 1
            elif self.__label[i] == label_predict[i] and self.__label[i] == 0:
                TN += 1
            elif self.__label[i] != label_predict[i] and self.__label[i] == 1:
                FN += 1
            else:
                FP += 1

        FPR = FP / (TN + FP) # also known as the fall out
        ACC = (TP + TN) / (TP + TN + FP + FN)
        F1_score = 2 * TP / (2 * TP + FP + FN)

        return FPR, ACC, F1_score






if __name__ == "__main__":
    plot_data0, train_data0 = data_generator(100, 0, 0.4)
    plot_data1, train_data1 = data_generator(100, 1, 0.4)
    label0 = [0 for i in range(100)]
    label1 = [1 for i in range(100)]
    label = label0 + label1
    train_data = train_data0 + train_data1
    plot_data_l = [plot_data0,plot_data1]


    lr = LR(train_data, label)
    # lr.gradAscent()
    # w = lr.get_w()
    # plot_data(plot_data_l, w, "gradient ascent" , 2, True)
    # lr.newton()
    # w = lr.get_w()
    # plot_data(plot_data_l, w, "Newton's method", 2, True)
    # lr.gradAscentReg()
    # w = lr.get_w()
    # plot_data(plot_data_l, w, "gradient ascent with regularization term", 2,True)
    # lr.newtonReg()
    # w = lr.get_w()
    # plot_data(plot_data_l, w, "Newton With Regularization term", 2, True)

    #generate a new set of data
    # plot_data0, train_data0 = data_generator(100, 0, 0.4, 1)
    # plot_data1, train_data1 = data_generator(100, 1, 0.4, 1)
    # label0 = [0 for i in range(100)]
    # label1 = [1 for i in range(100)]
    # label = label0 + label1
    # train_data = train_data0 + train_data1
    # plot_data_l = [plot_data0, plot_data1]
    #
    # lr2 = LR(train_data, label)
    # data = lr2.get_data()
    #
    # w = lr.get_w()
    # FPV, ACC, F1 = lr.extract_evalutation(data)
    # print()
    # print("Performance under a new set of data")
    # print("fall out: " + str(FPV))
    # print("accuracy:" + str(ACC))
    # print("F1 score:" + str(F1))
    # plot_data(plot_data_l, w, "Performance unser a new set of data", 2, True)

    # data, label = read_data()
    # uci_lr = LR(data, label)
    #uci_lr.gradAscent()
    # uci_lr.newton()
    #uci_lr.gradAscentReg()
    #uci_lr.newtonReg()





