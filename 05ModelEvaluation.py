#混淆矩阵
import numpy as np
from sklearn.metrics import confusion_matrix
y_pred = np.array([1,0,0,1,1,1])
y_true = np.array([0,1,1,0,0,1])
confusion_matrix(y_true,y_pred)
print(confusion_matrix)

(tn,fp,fn,tp) = confusion_matrix(y_true,y_pred).ravel()
print(tn,fp,fn,tp)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 读入数据
data = pd.read_table('./Data/data.txt', sep='\s+')
train_data, test_data, train_y, test_y = train_test_split(data.iloc[:,:-1], data.iloc[:,-1],test_size=.2, random_state=0)

# 训练模型并作出预测
lr_model = LogisticRegression()
lr_model.fit(train_data, train_y)
pred_y = lr_model.predict(test_data)
test_y = test_y.values
(tn,fp,tn,tp) = confusion_matrix(test_y,pred_y).ravel()

 
# 正确率
acc_rate = (tp+tn)/(tn+fn+fp+tp)

# 计算精度
precision_rate = tp/(tp+fp)

# 计算召回率
recall_rate = tp/(tp+fn)

# 计算F1值
f1_score = 2 * precision_rate * recall_rate/(precision_rate + recall_rate)

# 打印变量值
print('accuracy: ', acc_rate)
print('')
print('precision: ', precision_rate)
print('')
print('recall: ', recall_rate)
print('')
print('f1_score: ', f1_score)



#ROC & AUC
from sklearn.metrics import roc_curve
y = np.array([1,1,2,2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = roc_curve(y, scores, pos_label = 2)
print("fpr:", fpr)
print("tpr:", tpr)
print("thresholds:", thresholds)


import matplotlib.pyplot as plt

# 读入数据，训练模型，并进行预测
# 已训练
# 得到预测样本标签为1的概率列表
pred_prob = lr_model.predict_proba(test_data)
pred_prob_1 = [value[1] for value in pred_prob]

# 请在下方作答 #
# 配置图片大小
plt.figure(figsize=(20, 5))

# 声明绘制roc曲线的x轴，y轴数组
roc_x = []
roc_y = []

# 获取pred_prob_1的最值
min_prob = min(pred_prob_1)
max_prob = max(pred_prob_1)

## 生成大小为100的阈值数组，范围为[min_prob, max_prob]
threshold = np.linspace(min_prob,max_prob,100)



# 保存在不同阈值下所有的真正例和伪正例
all_tp = []
all_fp = []

# 从ndarray变量test_y中统计所有的正类标签class = 1的数量all_pos
all_pos = np.sum(test_y)
# 从ndarray变量test_y中统计所有的反类标签clss = 0的数量all_neg
all_neg = len(test_y) - all_pos

# 遍历阈值数组
for value in threshold:
    
    # 记录真正例和伪正例
    tp = 0
    fp = 0
    
    # 检查每一个预测样本的概率
    for index, prob in enumerate(pred_prob_1):
        # 概率prob大于阈值，则该样本被认为是正例
        
        if prob > value:
            # 使用条件判断语句：如果真实标签为1，真正例加1；如果真实标签为0，伪正例加1
            if test_y[index] == 1:
                tp = tp + 1
            elif test_y[index] == 0:
                fp = fp + 1
            
    
    # 保存入all_tp和allfp中
    all_tp.append(tp)
    all_fp.append(fp)
    
    # 假阳率 = fp/(fp + tn) 存入roc_x
    roc_x.append(fp/float(all_neg))
    # 真阳率 = tp / (tp + fn)存入roc_y
    roc_y.append(fp/float(all_pos))

# 绘制roc曲线
ax = plt.subplot(121)

# 使用plot函数
ax.plot(roc_x, roc_y)

# 曲线名称
plt.title('custom')

# 使用sklearn的函数绘制roc曲线
ax1 = plt.subplot(122)
fpr, tpr, threshold = roc_curve(test_y, pred_prob_1)
ax1.plot(fpr, tpr)
plt.title('sklearn')

# 展示绘制的ROC曲线
plt.show()