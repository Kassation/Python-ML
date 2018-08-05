from numpy import *
import pandas as pd

random.seed(6)

def distEuclidean(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


# 随机选择初始化质心
def randCent(dataArr, k):
    dim = dataArr.shape[1]

    # 创建用于存储质心矩阵的空变量
    centroids = mat(zeros((k, dim)))

    for j in range(dim):
        # 最小值
        minJ = min(dataArr[:, j])
        # 特征取值范围
        rangeJ = float(max(dataArr[:, j]) - minJ)

        # 某特征的随机k个值
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
        print(centroids)
    return centroids


# 读取数据，并转换为ndarray
data = pd.read_table('./Data/test_data_1.txt', sep='\t').values


# 调用函数得到随机质心
# centroids = randCent(data, 3)



# 请补全下方代码 #
def kMeans(dataArr, k):
    m = dataArr.shape[0]
    clusterAssment = mat(zeros((m, 2)))  # 创建存储簇分配结果的矩阵
    centroids = randCent(dataArr, k)  # 簇中心矩阵
    clusterChanged = True  # 簇分配结果改变标志
    while clusterChanged:
        clusterChanged = False

        # 1. 遍历每个数据点，计算其与各簇中心的距离，以最近簇作为分配结果
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):

                # 计算数据点i与簇中心j的欧式距离
                distJI = distEuclidean(centroids[j, :], dataArr[i, :])

                # 得到最小距离的簇中心minIndex及对应最小距离minDist
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j

            # 比较上一轮簇分配结果与当前促分配结果是否一致
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True

            # 更新样本簇分配结果
            clusterAssment[i, :] = minIndex, minDist ** 2

        # 2. 重新计算各簇的中心
        for cent in range(k):
            # 获得簇内数据矩阵
            ptsInClust = dataArr[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


# 对data进行聚类

centroids, clusterAssment = kMeans(data, 4)
print(centroids)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("notebook", font_scale=1.3)
sns.set_palette('Set2')


def plotResult(data, k):
    centroids, clusterAssment = kMeans(data, k)
    rect = [0.1, 0.1, 0.8, 0.8]
    fig = plt.figure()
    markers = ['s', 'o', '^', 'd']
    ax = fig.add_axes(rect)
    for cent in range(4):
        # 簇内数据矩阵和簇中心
        ptsInClust = data[nonzero(clusterAssment[:, 0].A == cent)[0]]
        ax.scatter(ptsInClust[:, 0],
                   ptsInClust[:, 1],
                   marker=markers[cent],
                   s=45)

    ax.scatter(centroids[:, 0].flatten().A[0],
               centroids[:, 1].flatten().A[0],
               marker='+',
               s=100,
               c='k')


plotResult(data, 4)

# sklearn 中的kmeans
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=6)
kmeans.fit(data)

centroids = kmeans.cluster_centers_
clusterAssment = kmeans.labels_
sse = kmeans.inertia_
print("centroids:\n", centroids)
print("label:\n", clusterAssment[:5])
print("sse:\n", sse)
