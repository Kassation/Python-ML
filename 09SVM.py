import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.svm import SVC

#读取数据
data = pd.read_csv("C://Users/zhang/Documents/Data-HackData/diabetes.csv")

#将目标特征与其他特征分离
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

#划分训练集train_X, train_y 和测试集test_X, test_y
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = .2, random_state = 0)

#训练集标准化，返回结果为scaled_train_X
scaler = preprocessing.StandardScaler()
scaler.fit(train_X)
scaled_train_X = scaler.transform(train_X)

#构建支持向量机
clf = SVC()

#模型训练
clf.fit(scaled_train_X, train_y)

#测试集标准化
scaled_test_X = scaler.transform(test_X)

#使用模型返回预测值
pred_y = clf.predict(scaled_test_X)

# 打印支持向量的个数，返回结果为列表，[-1标签的支持向量，+1标签的支持向量]
print(clf.n_support_)

# 使用classification_report函数进行模型评价
print(classification_report(test_y, pred_y))

# 构建惩罚系数为0.3的模型，并与之前的模型做比较
clf_new = SVC(C=0.3)
clf_new.fit(scaled_train_X, train_y)
pred_y = clf_new.predict(scaled_test_X)
print(clf_new.n_support_)
print(classification_report(test_y, pred_y))

#设置核函数
def rbf_kernel(X, Y, gamma=0.24):
# 获取X和Y的大小
    num1 = X.shape[0]
    num2 = Y.shape[0]
    
    # 计算X和X^T的矩阵乘积
    gram_1 = np.dot(X, X.T)
    
    # 获取gram_1对角线位置的元素，组成大小(num1, 1)的列表，并将整个列表复制，扩展为(num1, num2)大小的矩阵component1
    component1 = np.tile(np.diag(gram_1).reshape(-1, 1), (1, num2))
    
    # 计算Y和Y^T的乘积
    gram_2 = np.dot(Y, Y.T)
     
    # 获取gram_2对角线位置的元素，组成(1, num2)的列表，并将整个列表复制，扩展为(num1, num2)大小的矩阵component2
    component2 = np.tile(np.diag(gram_2).reshape(-1, 1).T, (num1, 1))
   
    # 计算2X和Y^T的内积 
    component3 = 2*np.dot(X, Y.T)

  
    # 返回结果
    result = np.exp(gamma*(- component1 - component2 + component3))
    return result

# 计算糖尿病患者训练数据集的核矩阵
rbf_matrix = rbf_kernel(scaled_train_X, scaled_train_X)

# 训练一个支持向量分类器
clf = SVC(kernel = rbf_kernel)
clf.fit(scaled_train_X, train_y)
pred_y = clf.predict(scaled_test_X)
print(clf.n_support_)
print(classification_report(test_y, pred_y))


rbf_clf = SVC(kernel = 'rbf', gamma = 0.24)
rbf_clf.fit(scaled_train_X, train_y)
pred_y = rbf_clf.predict(scaled_test_X)
print(rbf_clf.n_support_)
print(classification_report(test_y, pred_y))


#sklearn中的SVR(regression)
from sklearn.svm import SVR 
X = [[0,0], [1,-1]]
y = [0.2, 2.3]
clf = SVR()
clf.fit(X,y)

print(clf.predict([[1,1]]))