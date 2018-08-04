import numpy as np
import pandas as pd 

# 读取数据
weekend_data = pd.read_table("C://Users/zhang/Documents/Data-HackData/weekend.txt",  sep='\t')

## 自定义计算entropy的函数
def cal_entropy(data, feature_name):
    '''
    data : 数据集变量，DataFrame类型
    featrue_name : 目标特征名称
    '''
    # 声明数据集的熵
    entropy = 0

    #获取data的样本数num
    num = data.shape[0]

     ## 使用value_counts()函数获取目标特征`feature_name`取值的频数统计信息freq_stats
    freq_stats = data[feature_name].value_counts()
    
    ## 遍历目标特征的取值频数
    for index in range(len(freq_stats)):
        
        ## 获取具体的取值频数freq
        freq = freq_stats[index]
        
        ## 通过频数计算频率prob 
        prob = freq / float(num)
        
        ## 计算某个取值的entropy，
        entropy = entropy + (-prob*np.log2(prob))
    ## 返回结果
    return round(entropy, 3)

## 调用cal_entropy函数
data_entropy = cal_entropy(weekend_data, 'status')

## 自定义计算信息增益的函数cal_infoGain
## data: 数据集变量，DataFrame类型
## base_entropy: 数据集的熵
def cal_infoGain(data, base_entropy):
    
    ## 声明数据集特征的信息增益列表
    infogain_list = []
    
    ## 数据集关于特征的熵
    entropy_list = []
    
    ## 获取数据集的样本数nums, 维度dims
    nums, dims = data.shape
    
    
    ## 获取数据集的特征名称，类型为list
    feature_list = list(data.columns.values)

    ## 移除目标特征名称
    feature_list.remove('status')
    
    ## 遍历每个特征
    for feature in feature_list:
        
        ## 保存feature不同取值的加权熵
        sub_entropy = 0
        
        ## 切片数据集，获取特征feature的数据记录feature_data 
        feature_data = data[feature]
        
        ## 使用value_count()函数获取特征feature取值的统计信息freq_stats
        freq_stats = feature_data.value_counts()
        ## 获取freq_stats的索引信息，即是feature不同的取值，类型为ndarray
        value_stats = freq_stats.index.values
        ## 通过列表推导式计算feature各个取值出现的概率feature_prob
        feature_prob = [i/float(nums) for i in freq_stats]
        
        ## 遍历feature的每个取值以及取值出现的概率
        for pair_value in zip(value_stats, feature_prob):
            
            ## feature取值
            feature_value = pair_value[0]
            ## feature_value出现的概率
            feature_prob = pair_value[1]
            
            ## 对数据集data切片，获取特征feature取值为feature_value的数据样本sliced_data，类型为DataFrame
            sliced_data = data[feature_data == feature_value]
            ## 使用cal_entropy函数计算sliced_data的熵entropy
            entropy = cal_entropy(sliced_data, 'status')
            ## 累加featue取值的熵
            sub_entropy = sub_entropy + feature_prob*entropy
        
        ## 保存sub_entropy到entropy_list中
        entropy_list.append(sub_entropy)   
    
    ## 遍历entropy_list
    for value in entropy_list:
        
        ## 计算信息增益
        infogain = base_entropy - value
        ## infogain取值保留小数点后4位，保存到infogain_list中
        infogain_list.append(round(infogain, 4))
    
    ## 获取infogain_list的最大值max_infogain
    max_infogain = max(infogain_list)
    ## 获取最大值所在的位置索引max_index
    max_index = infogain_list.index(max_infogain)
    ## 根据max_index在feature_list中找到特征的名称best_feature
    best_feature = feature_list[max_index]
    ## 返回结果
    return infogain_list, best_feature

## 调用cal_infogain计算各个特征的信息增益infogains，并获取最优的分支节点名称best_feature
infogains, best_feature = cal_infoGain(weekend_data, data_entropy)

## 打印
print(u'信息增益列表：', infogains)
print('')
print(u'最优的分支节点名称：', best_feature)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv('C://Users/zhang/Documents/Data-HackData/product.csv')

## 构建分支节点选择方法为信息熵的决策树模型tree_model
tree_model = DecisionTreeClassifier(criterion='entropy')

## 对数据集切片，获取除了目标特征的其他特征的数据记录X
### 对数据集切片，获取目标特征`销量`的数据记录y
X = data.iloc[:, :-1]
y = data.iloc[:, -1]




## 使用train_test_split函数划分训练集train_X, train_y和测试集test_X, test_y
## 测试集所占比例为0.1,random_state为0
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = .1, random_state = 0)

## 拟合数据集
tree_model.fit(train_X, train_y)

## 使用tree_model对测试集test_X进行预测判别，获取预测的结果为pred_y
pred_y = tree_model.predict(test_X)

## 使用classification report函数对模型进行评价，并打印函数返回的结果
print(classification_report(test_y, pred_y))