import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./Data/user_review.csv')

# 查看data中的特征User continent的取值类型， 并打印输出的内容
print(data['User continent'].value_counts(dropna = False))
# 进行One-Hot编码
encode_uc = pd.get_dummies(data['User continent'], prefix = 'User continent_')
encode_uc.head()

# 查看data中的特征Traveler type的取值类型， 并打印输出的内容
print(data['Traveler type'].value_counts(dropna = False))

freq_v = 'Couples'

### 缺失值填充
data['Traveler type'] = data['Traveler type'].fillna(freq_v)

### 打印
print('')
print('缺失值填充完之后：')
print('')
print(data['Traveler type'].value_counts(dropna = False))


## 创建Z-score对象，并使用fit()方法
std_scaler = StandardScaler().fit(data[['Score']])

## 特征标准化，使用transform()方法
normal_df = std_scaler.transform(data[['Score']])

## 均值
score_mean = std_scaler.mean_

## 方差
score_var = std_scaler.var_

## 打印
print(score_mean)
print(score_var)

## 打印前五行内容
print(normal_df[:5])

# 特征离散化
## 使用Pandas的qcut()函数对data中的特征Member years进行等频离散化，结果保存在bins中；
bins = pd.qcut(data['Member years'],4)

## 统计取值信息
pd.value_counts(bins)

## 离群值检测
#使用拉依达准则对data的特征Member years进行离群值检测；
## Z-score标准化
member_data = data[['Member years']]
new_data = (member_data - np.mean(member_data))/np.std(member_data)

## 写出过滤条件
outlier_judge = abs(new_data['Member years']) >= 3

## 统计离群值的个数
outlier_num = sum(outlier_judge)

## 包含离群值的数据样本记录
outliers = data[abs(new_data['Member years']) >= 3]

print(outliers)