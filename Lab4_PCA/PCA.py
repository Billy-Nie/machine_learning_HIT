import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow.examples.tutorials.mnist.input_data as mnist_input_data
from PIL import Image

def data_generator():
    sampleNo = 10
    mux = 3
    sigmax = 1.5
    muy = 5
    sigmay = 1.5
    muz = 0
    sigmaz = 0.2
    x = np.random.normal(mux, sigmax, sampleNo)
    y = np.random.normal(muy, sigmay, sampleNo)
    z = np.random.normal(muz, sigmaz, sampleNo)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c = "red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    data = np.hstack((x, y))
    data = np.hstack((data, z))
    return data, x, y, z

def zero_mean(dataMat):
    '''
    按列求解该数据矩阵的平均值，然后所有数据减去对应的平均值
    以达到将平均值达到0的要求
    :param dataMat: 数据矩阵，每一行表示一个数据，每一列代表一个纬度（特征）
    :return: 平均值归零后的数据
    '''
    data_mean = np.mean(dataMat, axis=0) # 按照列求平均值，即求各个纬度的平均值
    data_zero_mean = dataMat - data_mean
    return data_zero_mean, data_mean
def coverience_matrix(data):
    m = data.shape[1]
    cov = np.zeros((m, m))
    for i in range(len(data)):
        #print(data[i].transpose().dot(data[i]))
        cov += data[i].reshape(1,-1).transpose().dot(data[i].reshape(1, -1))
        #cov += data[i].transpose().dot(data[i])
    cov = cov / m
    return cov

def rotate_data(data, eigVals, eigVects):
    '''
    根据特征向量和特征值选取k个纬度，然后进行数据旋转
    :param data:数据
    :param eigVals: 特征值
    :param eigVects: 特征向量
    :return: 旋转之后的数据
    '''
    def decide_k(eigVals, m, percent = 0.99):
        '''
        按照http://deeplearning.stanford.edu/wiki/index.php/主成分分析 （翻墙访问)
        中指定的公式来决定降维之后的纬度
        :param eigVals:特征值
        :param eigVects:特征向量
        :param m:降维之前的纬度
        :return:降维之后的数据纬度
        '''
        sum1 = sum(eigVals)
        eigVals_sort = np.sort(eigVals)
        eigVals_sort = eigVals_sort[::-1]
        sum2 = 0
        for i in range(m):
            sum2 += eigVals_sort[i]
            if sum2 / sum1 > percent:
                break
        return (i + 1)
    k = decide_k(eigVals, data.shape[1])
    print("k = " + str(k))
    eigValIndice = np.argsort(eigVals) #对特征值从小到大排列
    n_eigValIndice = eigValIndice[-1:-(k + 1):-1] # 最大k个特征值的下标
    U = []
    for i in range(k - 1):
        if i == 0:
            U = np.vstack((eigVects[n_eigValIndice[i]], eigVects[n_eigValIndice[i + 1]]))
        else:
            U = np.vstack((U, eigVects[n_eigValIndice[i + 1]]))
    data_rot = np.array(U).dot(data.transpose())
    return data_rot.transpose(), U

def restore_data(data_rot, U, data_mean):
    #print(data_rot.transpose().shape)
    #print(U.transpose().shape)
    restored_data = U.transpose().dot(data_rot.transpose()) + data_mean.reshape(-1, 1)
    return np.array(restored_data)


def PCA(dataMat):
    data, data_mean = zero_mean(dataMat) #将所有数据的均值归一化到零附近
    cov = coverience_matrix(data) #计算协方差矩阵
    eigVals, eigVects = np.linalg.eig(np.mat(cov)) #求特征值和特征向量
    data_rot, U = rotate_data(data, eigVals, eigVects) # 旋转、降维
    restored_data = restore_data(data_rot, U, data_mean) # 恢复数据
    return data_rot, restored_data, U # 返回降维后的和恢复之后的数据以及旋转用的矩阵U以备用

def mnist():
    mnist = mnist_input_data.read_data_sets("MNIST_data/", one_hot=False)
    imgs = mnist.train.images
    labels = mnist.train.labels
    # print(type(imgs)) # <class 'numpy.ndarray'>
    # print(type(labels)) # <class 'numpy.ndarray'>
    # print(imgs.shape) # (55000, 784)
    # print(labels.shape) # (55000,)
    #提取前1000长照片中的100个数字7
    origin_7_imgs = []
    for i in range(1000):
        if labels[i] == 7 and len(origin_7_imgs) < 100:
            origin_7_imgs.append(imgs[i])
    # print(np.array(origin_7_imgs).shape) # (100, 784)
    def array_to_image(array):
        '''
        将这个784长度的一维数组转化成图像
        '''
        array = array * 255
        new_img = Image.fromarray(array.astype(np.uint8))
        return new_img

    def comb_imgs(origin_imgs, col, row, each_width, each_height, new_type):
        new_img = Image.new(new_type, (col * each_width, row * each_height,))
        for i in range(len(origin_imgs)):
            each_img = array_to_image(np.array(origin_imgs[i]).reshape(each_width, each_width))
            new_img.paste(each_img, ((i % col) * each_width, (i // col) * each_width))
        return new_img

    hundred_origin_7_imgs = comb_imgs(origin_7_imgs, 10, 10, 28, 28, 'L')
    hundred_origin_7_imgs.show()

    low_d_feat_for_7_imgs, restored_imgs, U = PCA(np.array(origin_7_imgs))
    low_d_img = comb_imgs(restored_imgs.transpose(), 10, 10, 28, 28, 'L')
    low_d_img.show()


if __name__ == "__main__":
    dataMax,x, y, z = data_generator()
    data, data_mean = zero_mean(dataMax)
    cov = coverience_matrix(data)

    eigVals,eigVects=np.linalg.eig(np.mat(cov)) #求特征值和特征向量
    data_rot, U = rotate_data(data, eigVals, eigVects)
    plt.scatter(*data_rot.transpose())
    plt.title("rotated data")
    plt.show()
    restored_data = restore_data(data_rot, U, data_mean)
    #_,restored_data,_ = PCA(dataMax)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c="red", label = "generated data")
    ax.scatter(restored_data[0], restored_data[1], restored_data[2], c = "blue", label = "PCA restored data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.legend()
    plt.show()

    mnist()


