import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
n_components = 590

# 读取数据
iris = pd.read_csv('./Data/iris.csv', usecols=range(4))


# 定义主成分分析函数
def pca(dataMat, n_components):
    # 数据中心化
    meanVals = np.mean(dataMat, axis=0)
    centered = dataMat - meanVals

    # 计算协方差矩阵，并进行特征值分解
    # 注意，协方差矩阵计算需指定rowvar参数为False，即表明每一列
    # 为每一个特征的不同取值
    covMat = np.cov(centered, rowvar=False)
    w, v = LA.eig(covMat)

    # 将特征值从大到小排序，并得到对应的索引值序列
    idxMin2Max = np.argsort(w)
    idxMax2Min = idxMin2Max[::-1]

    # 取前n_components个特征向量构成转换矩阵
    vidx = idxMax2Min[:n_components]
    tfMat = v[:, vidx]

    # 将原始数据转换到新的低维空间
    lowdMat = np.matmul(dataMat, tfMat)

    return lowdMat, tfMat, w, v


# 使用自定义pca函数对鸢尾花数据进行降维
#irisMat = np.mat(iris)
#iris2d, _ = pca(irisMat, n_components=2)

secondata = pd.read_csv('./Data/secon.csv', header = None)
split = np.array(secondata.iloc[0,0].split())
map(lambda x: float(x), split)
secon = pd.DataFrame(split).T
for i in range(1,secondata.shape[0]):
    split = np.array(secondata.iloc[i,0].split())
    map(lambda x: float(x), split)
    secon = secon.append(pd.DataFrame(split).T)
for i in range(secon.shape[0]):
    for j in range(secon.shape[1]):
        if secon.iloc[i,j] == 'NaN':
            secon.iloc[i,j] = np.nan

secon.fillna(method = 'ffill')

seconMat = np.mat(secon)

# pca降维
lowdMat, tfMat, w, v = pca(seconMat, n_components = n_components)
# 计算第一主成分占比
fpc = w[0]/sum(w)

# 每一个主成分的方差占比和累积方差占比计算
vp = map(lambda x:x/sum(w),w)
accvp = np.cumsum(vp)

# 绘制主成分方差占比随主成分数目的变化图
# 从图可以看出，为不损失太多信息，可选择保留前6个主成分
f,ax = plt.subplots(figsize = (6,6))
plt.plot(range(1,591), accvp, 'go-')
plt.xlabel('# of pricipal components')
plt.ylabel('percentage of accumalated variance')
plt.xticks(range(1,591,1))
plt.xlim([0,20])

plt.show()
# 计算重构误差
meanVal = np.mean(seconMat, axis=0)  # 均值
reconMat = lowdMat.dot(tfMat.T) + np.array(meanVal) # 重构得到的数据
errMat = np.array(secon) - reconMat # 重构误差
froerr = LA.norm(errMat, 'fro') # Frobenius范数计算