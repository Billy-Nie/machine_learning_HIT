import numpy as np
import matplotlib.pyplot as plt
import random

def data_generator(k = 5):
    '''
    generate 5 classes of data for the algorithm
    :return: 5 classes of data
    '''
    sample_num = 100;

    mu1 = np.array([[1, 1]])
    sigma1 = np.array([[0.8, 0],[0, 0.8]])

    mu2 = np.array([[10,4]])
    sigma2 = np.array([[1, 0], [0, 1]])

    mu3 = np.array([[6, 5]])
    sigma3 = np.array([[0.5, 0], [0, 2]])

    mu4 = np.array([[3, 4]])
    sigma4 = np.array([[0.3, 0],[0, 0.5]])

    mu5 = np.array([[5, -1]])
    sigma5 = np.array([[0.8, 0],[0, 0.8]])

    s1 = np.random.multivariate_normal(mu1[0], sigma1, sample_num)
    plt.scatter(*s1.T, label="class 1")
    s2 = np.random.multivariate_normal(mu2[0], sigma2, sample_num)
    plt.scatter(*s2.T, label="class 2")
    s1 = np.concatenate((s1, s2), axis = 0)
    s3 = np.random.multivariate_normal(mu3[0], sigma3, sample_num)
    plt.scatter(*s3.T, label="class 3")
    s1 = np.concatenate((s1, s3), axis=0)
    s4 = np.random.multivariate_normal(mu4[0], sigma4, sample_num)
    plt.scatter(*s4.T, label="class 4")
    s1 = np.concatenate((s1, s4), axis = 0)
    s5 = np.random.multivariate_normal(mu5[0], sigma5, sample_num)
    plt.scatter(*s5.T, label="class 5")
    s1 = np.concatenate((s1, s5), axis = 0)

    plt.legend()
    plt.title("Answers to the data generated")
    plt.show()
    return s1

def Euclidean_distance(point0, point1, n = 2): # n表示数据的维度，默认为2维
    distance = 0
    for i in range(n):
        distance += (point0[i] - point1[i]) ** 2
    return distance ** 0.5

def assigh_class(data, center_l,train_time, n = 2): # n表示数据的维度，train_time表示这是第几次训练
    for i in data:
        distances = [Euclidean_distance(i, j, n) for j in center_l]
        min_distance = min(distances)
        center = distances.index(min_distance) + 1
        i[2] = center
    trainning_plot(data, center_l, train_time)
    return data

def reassign_classes(data, center_l,n = 2):
    #根据data的分组信息重新计算数据中心
    #n表示数据的维数
    N = len(center_l)
    center_l_new = [[0 for i in range(n)] for i in range(N)]
    number_in_center = [0 for i in range(N)]
    for i in data:
        center = int(i[2] - 1)
        for j in range(n):
            center_l_new[center][j] += float(i[j])
        number_in_center[center] += 1
    for i in range(N):
        for j in range(n):
            center_l_new[i][j] /= number_in_center[i]
    distance = []
    for i in range(N):
        distance.append(Euclidean_distance(center_l_new[i], center_l[i]))
    return center_l_new, np.array(distance)


def trainning_plot(data, center_l, train_time = 0): # train_time表示是训练的第几次
    N = len(center_l)
    cmap = plt.cm.get_cmap("hsv", N + 1)
    rearange_data = np.array([])
    for i in range(N):
        plt.scatter(center_l[i][0], center_l[i][1],s = 300, c = cmap(i), marker="*", label= "class center " + str(i+1))
        type = list(filter(lambda x: x[2] == (i + 1), data))
        if i == 0:
            rearange_data = np.array([j for j in type])
        else:
            rearange_data = np.concatenate((rearange_data, np.array([j for j in type])))
        type = np.array([np.delete(j, -1) for j in type])
        if len(type) > 0:
            plt.scatter(*type.T, c = cmap(i))
    plt.legend()
    plt.title("trainning plot " + str(train_time))
    plt.show()
    return rearange_data




def init_classes(data, k = 5):
    '''
    从数据集data中随机选取k个点当作第一轮的数据中心
    :param data: 数据集
    :param k: 一共有k个数据中心
    :return: 一个k维的list，每一个元素对应一个数据中心
    '''
    n = len(data)
    class_l = []
    for i in range(k):
        index = np.random.randint(0, n)
        class_l.append(list(data[index]))
    return class_l

if __name__ == "__main__":
    s = data_generator()
    data = np.array([np.append(s[i], 0) for i in range(len(s))]) #0 means the data has not been assigned a value
    class_l = init_classes(s)
    training_time = 1

    data = assigh_class(data, class_l, training_time, 2)
    class_l, distance = reassign_classes(data, class_l)
    true_l = distance > 0.05
    print(distance)
    while(True in true_l):
        training_time += 1
        data = assigh_class(data, class_l, training_time, 2)
        class_l, distance = reassign_classes(data, class_l)
        true_l = distance > 0.005
        print(distance)

