import pandas as pd
import numpy as np
train_data = pd.read_table("./Data/gender.txt", sep = '\t')
test_data = pd.read_table("./Data/test.txt", sep = '\t')

print(u'训练集大小：', train_data.shape)
print(' ')
print(u'测试集大小：', test_data.shape)

#欧氏距离计算

def cal_eculidean(array_a, array_b):
    
    ## 计算变量各个维度的差的平方
    array_square = (array_a - array_b) ** 2
    
    ## 计算差的平方和array_square_sum
    array_square_sum = np.sum(array_square)
    
    ## 开根号
    sum_root = np.sqrt(array_square_sum)
    
    return sum_root

## 声储存距离的变量
distances = []

## 遍历`test_data`中的每个数据样本, 样本的储存格式为ndarray
for item in test_data.iloc[:,:-1].values:
    
    ## 储存数据样本和训练集中每个样本的距离，dist格式为字典类型，形如{index: distance}
    dist = {}
    
    for index, train_item in enumerate(train_data.iloc[:,:-1].values):
        
        ## 使用自定义函数计算item和train_item之间的举例distance
        distance = cal_eculidean(item, train_item)
        ## 将distance，和train_item的index存入字典变量dist中
        dist[index] = distance
    
    #将dist加入到distances中
    distances.append(dist)

#选择最近邻样本

## 声明储存K个邻居的index的列表
index_list = []

## 遍历每个测试样本的与训练集的距离，item类型为dict
for item in distances:
    ## item的类型为dict，即{'index': distance}
    ## 使用sorted函数对item按照distance进行升序排列,并取前10个距离最小的样本
    sorted_item = sorted(item.items(), key=lambda x: x[1])[:10]

    ## sorted_item形如[(index, distance),(index, distance)...(index, distance)]
    ## 使用zip函数从sorted_items中获取对应的index
    indexes = list(zip(*sorted_item))[0]
    ## 将获取的indexes加入到index_list中
    index_list.append(indexes)

#print(index_list)

## 声明预测类别数组
pred_label = []

## 遍历索引数组
for indexes in index_list:
    
    ## 获取测试样本K个邻居的数据记录neighbor_df,类型为Dataframe
    neighbor_df = train_data.loc[indexes, :]
    
    ## 找到出现频率最高的类别mode_value，类型为str
    mode_value = neighbor_df['gender'].mode().loc[0]
    
    ## 将mode_value保存在pred_label
    pred_label.append(mode_value)
    
## 将取值作变换 ‘M’:1  ‘F’:0
pred_label = [1 if item == 'M' else 0 for item in pred_label]

## 预测值与真实值相减，获得结果diff
diff = test_data['gender'].values - pred_label

## 计算正确率
accuracy_rate = len(diff[diff == 0])/float(len(diff))
print(accuracy_rate)





import matplotlib.pyplot as plt

## 声明保存预测错误样本个数列表
error = []

from sklearn import neighbors

X = train_data.iloc[:,:-1].values
y = [1 if item == 'M' else 0 for item in train_data.iloc[:,-1].values]

## K的取值范围
k_value = range(3, 20, 1)

## 遍历K = 3，4，5，6 ... 20
for num in k_value:
    ## 调用knn_classify函数，返回预测结果pred
    knn = neighbors.KNeighborsClassifier(num+1, weights = 'uniform')
    knn.fit(X, y)

    ## 预测结果转换 ‘M’:1 'F':0
    y_test = test_data.iloc[:,:-1].values
    y_pred = knn.predict(y_test)
    #y_pred = [1 if item == 'M' else 0 for item in y_pred]

    ## 预测值与真实值之查
    diff = y_pred - test_data['gender'].values

    ## 统计预测错误的样本个数
    error_num = len(diff[diff != 0])

    ## 将具体的K值对应的样本个数保存到列表error
    error.append(error_num)

## 计算正确率
rate_list = [1 - float(item) / len(diff) for item in error]

## 绘制折线图
plt.plot(k_value, rate_list, '--r*')
## 输出图像
plt.show()