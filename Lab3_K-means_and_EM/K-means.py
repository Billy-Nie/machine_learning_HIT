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

    mu3 = np.array([[-2, 5]])
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
        i[n] = center
    trainning_plot(data, center_l, train_time)
    return data

def reassign_classes(data, center_l,n = 2):
    #根据data的分组信息重新计算数据中心
    #n表示数据的维数
    N = len(center_l)
    center_l_new = [[0 for i in range(n)] for i in range(N)]
    number_in_center = [0 for i in range(N)]
    for i in data:
        center = int(i[n] - 1)
        for j in range(n):
            center_l_new[center][j] += float(i[j])
        number_in_center[center] += 1
    for i in range(N):
        for j in range(n):
            center_l_new[i][j] /= number_in_center[i]
    distance = []
    for i in range(N):
        distance.append(Euclidean_distance(center_l_new[i], center_l[i], n))
    return center_l_new, np.array(distance)


def trainning_plot(data, center_l, train_time = 0): # train_time表示是训练的第几次
    N = len(center_l)
    cmap = plt.cm.get_cmap("hsv", N + 1)
    rearange_data = np.array([])
    for i in range(N):
        type = list(filter(lambda x: x[2] == (i + 1), data))
        if i == 0:
            rearange_data = np.array([j for j in type])
        else:
            rearange_data = np.concatenate((rearange_data, np.array([j for j in type])))
        type = np.array([np.delete(j, -1) for j in type])
        if len(type) > 0:
            plt.scatter(*type.T, c = cmap(i))
        plt.scatter(center_l[i][0], center_l[i][1], s=300, marker="*", label="class center " + str(i + 1))
    plt.legend()
    plt.title("trainning plot " + str(train_time))
    plt.show()
    return rearange_data




def init_classes(data, k = 5, j = 2):
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
        if i  > 0:
            while Euclidean_distance(data[index], class_l[-1], j) < 2:
                index = np.random.randint(0, n) # 防止两个点距离太近
        class_l.append(list(data[index]))

    return class_l

def iris_data_reader():
    file = open("uci_data/iris.data", "r")
    lines = file.readlines()
    data = []
    answer = []
    for line in lines:
        line.strip()
        line_sp = line.split(",")
        line_data = []
        for i in range(len(line_sp) - 1):
            line_data.append(float(line_sp[i]))
        data.append(line_data)
        if line_sp[-1] == "Iris-setosa\n":
            temp_line_data = line_data[:]
            temp_line_data.append("Iris-setosa")
        elif line_sp[-1] == "Iris-versicolor\n":
            temp_line_data = line_data[:]
            temp_line_data.append("Iris-versicolor")
        else:
            temp_line_data = line_data[:]
            temp_line_data.append("Iris-virginica")
        answer.append(temp_line_data)
    return data, answer

def similarity(class_i, class_l, k = 4):
    #只用在uci的iris数据集合中，它比较我们预测得到的集合类i和答案中的三个集合"Iris-setosa", "Iris-versicolor", "Iris-virginica"，找到哪一个集合和我们这个最像，
    #然后将我们这个集合里面所有的类i换成相应的字符串"Iris-setosa"或"Iris-versicolor"或"Iris-virginica"
    n = len(class_l)
    class_number = [0 for i in range(n)]
    for i in class_i:
        for j in range(n):
            for l in range(len(class_l[j])):
                result = True
                for m in range(k):
                    a = class_l[j][l][m]
                    b = i[m]
                    result = result and (i[m] == class_l[j][l][m])
                if result == True:
                    class_number[j] += 1

    return class_number.index(max(class_number))

if __name__ == "__main__":
    # s, answer = iris_data_reader()
    # data = []
    # for i in range(len(s)):
    #     data.append(s[i][:])
    #     data[i].append(0)

    s = data_generator()
    data = np.array([np.append(s[i], 0) for i in range(len(s))]) #0 means the data has not been assigned a value
    class_l = init_classes(s, 5, 2)
    training_time = 1

    data = assigh_class(data, class_l, training_time, 2)
    class_l, distance = reassign_classes(data, class_l, 2)
    true_l = distance > 0.05
    print(distance)
    while(True in true_l):
        training_time += 1
        data = assigh_class(data, class_l, training_time, 2)
        class_l, distance = reassign_classes(data, class_l, 2)
        true_l = distance > 0.005
        print(distance)


    # print("答案：")
    # print(answer)
    # print("*" * 23 )
    #
    #
    # class_1 = list(filter(lambda x: x[4] == 1, data))
    # class_2 = list(filter(lambda x: x[4] == 2, data))
    # class_3 = list(filter(lambda x: x[4] == 3, data))
    #
    # class_setosa = list(filter(lambda x: x[4] == "Iris-setosa", answer))
    # class_versicolor = list(filter(lambda x: x[4] == "Iris-versicolor", answer))
    # class_virginica = list(filter(lambda x: x[4] == "Iris-virginica", answer))
    # class_l = [class_setosa, class_versicolor, class_virginica]
    #
    # uci_class = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    # a = similarity(class_1, class_l, 4)
    # b = similarity(class_2, class_l, 4)
    # c = similarity(class_3, class_l, 4)
    # for i in data:
    #     if i[4] == 1:
    #         i[4] = uci_class[a]
    #     elif i[4] == 2:
    #         i[4] = uci_class[b]
    #     else:
    #         i[4] = uci_class[c]
    # error = 0
    # for i in range(len(data)):
    #     if data[i][4] != answer[i][4]:
    #         error += 1
    #
    # print("预测")
    # print(data)
    # print("错误率 : " + str(error / len(data)))



