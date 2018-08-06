import pandas as pd
import numpy as np
from sklearn import linear_model
insurance = pd.read_csv('./Data/insurance.csv')
age = insurance['age'].values
charges = insurance['charges'].values

# 定义一元线性回归函数
def linearRegression(xArr,yArr):
    
    # x均值,y均值计算
    mean_x = xArr.mean()
    mean_y = yArr.mean()
    numerator = sum(xArr * yArr)-len(xArr)*mean_x*mean_y
    denominator = sum(xArr * xArr) - len(xArr)*mean_x*mean_x
    # w0,w1计算
    w1 = float(numerator)/denominator
    w0 = mean_y - w1 * mean_x
    return round(w0,2),round(w1,2)
    
# 模型训练，得到参数值
w0, w1 = linearRegression(age, charges)
print(w0)
print(w1)

# sklearn的训练结果
regr = linear_model.LinearRegression()
regr.fit(age.reshape(-1,1), charges.reshape(-1,1))
print(round(regr.coef_[0][0],2))
print(round(regr.intercept_[0],2))


# 定义多元线性回归函数
def linearRegression(xArr, yArr):
    # 将Array转换为矩阵
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    # 判定协方差矩阵是否可逆
    if np.linalg.det(xTx) == 0.0:
        print("singular matrix, cannot do inverse")
    # 计算参数向量
    ws = np.linalg.solve(xTx, xMat.T * yMat)
    return ws


# 模型训练，得到参数值
X = insurance[['age', 'bmi', 'children']].values
X = np.column_stack((X, np.ones(X.shape[0])))
y = insurance['charges']
ws = linearRegression(X, y)

# sklearn的训练结果
regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.coef_)
print(regr.intercept_)

#one-hot特征编码
encode_sex = pd.get_dummies(insurance['sex'], prefix = 'sex_')
encode_smk = pd.get_dummies(insurance['smoker'], prefix = 'smoker_')
encode_region = pd.get_dummies(insurance['region'], prefix = 'region_')

# 根据X、y和自定义函数linearRegresssion训练模型参数ws，并计算预测值y_pred
X = insurance.drop(['charges'], axis=1).drop(['sex'], axis=1).drop('smoker',axis=1).drop(['region'],axis=1)
X = X.join(encode_smk).join(encode_region).join(encode_sex)
y = insurance['charges']

ws = linearRegression(X,y)
y_pred = np.mat(X.values)*ws
y_pred = np.array(y_pred).reshape(y_pred.shape[0],) # 将矩阵转换为一行多列的array格式

# 自定义决定系数函数，并对训练得到的模型进行评价
def r2_Score(y_true, y_pred):
    
    # 计算SST,SSR或SSE
    sst = sum((y_true - np.mean(y_true))**2)
    ssr = sum((y_true - y_pred)**2)
    sse = sum((y_pred - np.mean(y_true))**2)
    
    # 根据SST,SSR或SSE计算决定系数r2
    r2 = 1 - float(ssr)/sst

    return round(r2,2)

# 根据y和y_pred计算决定系数score
score = r2_Score(y, y_pred)
print("score1:" , score)

# sklearn模型训练结果
from sklearn import linear_model, metrics
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(X, y)
y_pred = regr.predict(X)
score_sklearn = round(metrics.r2_score(y, y_pred), 2)
print("r2-Score:", score_sklearn)