import numpy as np
import matplotlib.pyplot as plt
import random
from Kmeans import Kmeans, iris_data_reader, similarity, Euclidean_distance
from scipy.stats import multivariate_normal

"""
Generate a two-dimensional Gaussian Mixture Model
with three gaussian distributions linearly added
"""
def data_gernerator():
    sampleNum = 1500
    #Gaussian 1
    mu1 = np.array([[1, 5]])
    Sigma1 = np.array([[0.1, 0], [0, 2]])
    s1 = np.random.multivariate_normal(mu1[0], Sigma1, 1)
    #Gaussian 2
    mu2 = np.array([[-1, -2]])
    Sigma2 = np.array([[1, 0], [0, 1.5]])
    s2 = np.random.multivariate_normal(mu2[0], Sigma2, 1)
    #Gaussian 3
    mu3 = np.array([[4, 2]])
    Sigma3 = np.array([[0.8, 0], [0, 0.8]])
    s3 = np.random.multivariate_normal(mu3[0], Sigma3, 1)

    for i in range(sampleNum):
        if(random.random() < 0.5):
            s = np.random.multivariate_normal(mu1[0], Sigma1, 1)
            s1 = np.concatenate((s1, s), axis=0)
        elif(0.5 <= random.random() <= 0.8):
            s = np.random.multivariate_normal(mu2[0], Sigma2, 1)
            s2 = np.concatenate((s2, s), axis=0)
        else:
            s = np.random.multivariate_normal(mu3[0], Sigma3, 1)
            s3 = np.concatenate((s3, s), axis=0)


    plt.scatter(*s1.T)
    plt.scatter(*s2.T)
    plt.scatter(*s3.T)
    plt.title("GMM data set")
    plt.show()
    s = np.concatenate((s1, s2, s3), axis=0)
    return s

def EM(s, k, n, m = 2):
    '''
    使用EM算法来估计隐变量的值
    :param s: 数据集s
    :param k: 混合高斯模型由几个高斯先行组合而成
    :param n: 数据集的大小
    :param m: 每一个数据的纬度
    :return: 一个n * k纬的矩阵，每一行中的k个数据分别对应估计来自第i类的概率
    '''

    #对k个类进行初始化pi，mu，Sigma矩阵
    pi = []
    Sigma = []

    #使用Kmeans算法对该数据集进行mu的初始化
    data = np.array([np.append(s[i], 0) for i in range(len(s))])
    mu, data = Kmeans(s, data, k, m)
    for i in range(k):
        data_filtered = (list(filter(lambda x: x[m] == (i+1), data)))
        data_filtered = [list(i)[:-1] for i in data_filtered]
        data_filtered = np.array(data_filtered)
        Sigma_i = np.cov(data_filtered.T)
        if i < k-1:
            pi.append(random.random())
        else:
            pi_sum = sum(pi)
            pi.append(1 - pi_sum)
        Sigma.append(Sigma_i)


    #************************EM*********************************
    while True:
        Gamma = E_step(k, n, s, mu, Sigma, pi)
        mu_new, Sigma_new, pi_new = M_step(k, n, Gamma, s, m)
        if sum([(pi_new[i] - pi[i]) ** 2 for i in range(k)]) ** 0.5 < 0.0000001:
            break
        mu = mu_new
        Sigma = Sigma_new
        pi = pi_new
        print(pi_new)
    print("mu:")
    print(mu)
    print("Sigma:")
    print(Sigma)
    print("Gamma:")
    print(Gamma)
    return Gamma





def E_step(k, n, s, mu, Sigma, pi):
    #k表示由多少个高斯分布线性组合而成
    #n表示共有多少数据
    #s表示数据
    #mu表示均值向量
    #Sigma表示协方差矩阵
    #pi表示每一个高斯占比是多少
    Gamma = [[0 for i in range(k)] for i in range(n)]

    for i in range(n):
        sum = 0
        for j in range(k):
            # print(Sigma[j])
            # print(s[i])
            # print(mu[j])
            # print("************************************************")
            sum += pi[j] * multivariate_normal.pdf(s[i], mu[j], Sigma[j])
        for j in range(k):
             gammaNK = pi[j] * multivariate_normal.pdf(s[i], mu[j], Sigma[j]) / sum
             Gamma[i][j] = gammaNK
    return Gamma

def M_step(k, n, Gamma, s, m):
    #k表示由多少个高斯分布线性组合而成
    #n表示共有多少个数据
    #Gamma表示后验概率的数组
    #s表示数据
    #m表示数据的纬度
    #Sigma表示协方差矩阵的list
    mu_new = [[0 for i in range(m)] for j in range(k)]
    Sigma_new = []
    pi_new = []
    for i in range(k):
        Nk = 0
        for j in range(n):
            Nk += Gamma[j][i]
        mu_k = [0 for i in range(m)]
        for j in range(n):
            mu_k = [mu_k[k] + Gamma[j][i] * s[j][k] for k in range(m)]
        mu_k = [i / Nk for i in mu_k]
        mu_new[i] = mu_k
        mu_k_nd = np.array(mu_k).reshape(-1, 1)

        sigma = np.array([[0.0 for j in range(m)] for i in range(m)])
        for j in range(n):
            sn_nd = np.array(s[j]).reshape(-1, 1)
            sigma += Gamma[j][i] * (sn_nd - mu_k_nd).dot((sn_nd - mu_k_nd).transpose())
        sigma = sigma/Nk
        sigma = [list(sigma[k]) for k in range(sigma.shape[0])]
        Sigma_new.append(sigma)

        pi_k = Nk / n
        pi_new.append(pi_k)

    return mu_new, Sigma_new, pi_new



def plot_result(Gamma, s, n):
    '''
    画出EM算法给出的最后结果
    :param Gamma: 后验概率
    :param s:数据集
    :param n:数据集的大小
    :return: 在屏幕上输出一个图，其中每一个点的颜色是由3原色中的按照不同比例混合而来，这个比例就是Gamma中推测的来自每一个类型的概率
    '''
    for i in range(n):
        plt.scatter(s[i][0], s[i][1], c = Gamma[i])
    plt.title("EM output")
    plt.show()

if __name__ == "__main__":
    print("*****************生成数据集*************************")
    s = data_gernerator()
    Gamma = EM(s, 3, len(s), 2)
    plot_result(Gamma, s, len(s))

    print("*****************uci鸢尾花数据集************************")
    s, answer = iris_data_reader()
    data = np.array([np.append(s[i], 0) for i in range(len(s))])

    Gamma = EM(s, 3, len(s), len(s[0]))
    class_setosa = list(filter(lambda x: x[4] == "Iris-setosa", answer))
    class_versicolor = list(filter(lambda x: x[4] == "Iris-versicolor", answer))
    class_virginica = list(filter(lambda x: x[4] == "Iris-virginica", answer))
    class_l = [class_setosa, class_versicolor, class_virginica]

    class_1 = []
    class_2 = []
    class_3 = []
    for i in range(len(s)):
        class_i = max(Gamma[i])
        class_i = Gamma[i].index(class_i) + 1
        if class_i == 1:
            class_1.append(np.append(s[i], 1))
        if class_i == 2:
            class_2.append(np.append(s[i], 2))
        else:
            class_3.append(np.append(s[i], 3))
        #print(class_i)


    uci_class = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    a = similarity(class_1, class_l, 4)
    b = similarity(class_2, class_l, 4)
    c = similarity(class_3, class_l, 4)

    class_l  = []
    class_l.append(class_1)
    class_l.append(class_2)
    class_l.append(class_3)

    for i in range(len(class_l)):
        for j in range(len(class_l[i])):
            class_l[i][j] = list(class_l[i][j])

    for i in range(len(class_l)):
        if i == 0:
            for j in class_1:
                j[4] = uci_class[a]
        elif i == 1:
            for j in class_2:
                j[4] = uci_class[b]
        else:
            for j in class_3:
                j[4] = uci_class[c]

    right = 0
    right_l = []
    flag = False
    for i in class_l:
        right = 0
        if i[0][4] == "Iris-setosa":
            for j in range(len(i)):
                flag = False
                for k in range(len(class_setosa)):
                    if Euclidean_distance(i[j], class_setosa[k], 4) == 0 and not flag:
                        right += 1
                        flag = True
            print("正确率setosa：")
            print(right/len(class_setosa))
        elif i[0][4] == "Iris-versicolor":
            for j in range(len(i)):
                flag = False
                for k in range(len(class_versicolor)):
                    if Euclidean_distance(i[j], class_versicolor[k], 4) == 0 and not flag:
                        right += 1
                        flag = True
            print("正确率versicolor:")
            print(right / len(class_versicolor))
        else:
            for j in range(len(i)):
                flag = False
                for k in range(len(class_virginica)):
                    if Euclidean_distance(i[j], class_virginica[k], 4) == 0 and not flag:
                        right += 1
                        flag = True
            print("正确率virginica:")
            print(right / len(class_virginica))




