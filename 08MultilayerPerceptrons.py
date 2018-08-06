import numpy as np

# 输入X和y
X = np.array([ [0,0,1],
[1,1,1],
[1,0,1],
[0,1,1]])

y = np.array([[0,1,1,0]]).T

# 请在下方作答 #
# Sigmoid激活函数以及其导数
def sigmoid(x, derivative = False):
    # 计算sigmoid的输出
    sigmoid_value = 1./(1 + np.exp(-x))
    if derivative == False:     
        return sigmoid_value
    
    elif derivative == True:
        # 计算sigmoid的导数
        return sigmoid_value*(1 - sigmoid_value)

# 迭代次数
iter_num  = 1000

# 初始化权重向量w
num, dim = X.shape
w_vec = np.ones((dim, 1))

for iter in xrange(iter_num):
    
    ## 通过权重向量w_vec，实现线性加和，结果为z1
    z_1 = np.dot(X, w_vec)
    
    # 经过激活函数Sigmoid，获得输出a_1
    a_1 = sigmoid(z_1)
      
    # 模型输出a_1与真实值的误差
    error = a_1 - y
    
    # 已知损失函数loss为交叉熵即sum (ylog a_1 + (1-y)log (1 - a1)), 计算损失函数针对w_vec的梯度
    # w_vec_delta = np.dot((d_z_1 / d_w_vec),(np.multiply((d_loss / d_a_1),(d_a1 / d_z_1)))) / num
    w_vec_delta = np.dot(X.T, error) / float(num)
    
    # 更新权重
    w_vec = w_vec - 0.2*w_vec_delta   
print(w_vec)


#多层感知机
#数据集X未知
## 数据集的样本数num，和维度dim
dim, num = X.shape

## 迭代次数
iter_num = 10000

## 隐含层的神经元数
hidden_num = 4

## 初始化权重矩阵，不考虑偏置项b
np.random.seed(3)
w_0 = np.random.random((hidden_num, dim)) 
w_1 = np.random.random((hidden_num, 1)).T


for j in xrange(iter_num):
    
    ##正向激励传播：z_1  a_1, z_2, a_2
    ##隐含层输入z_1 大小为(hidden_num, num)
    z_1 = np.dot(w_0, X)
    
    ##隐含层输出a_1, 大小为(hidden_num, num)
    a_1 = sigmoid(z_1)
    
    ##输出层输入z_2，大小为(1, num)
    z_2 = np.dot(w_1, a_1)
    
    ##输出层输出a_2，大小为(1, num)，即是模型预测值
    a_2 = sigmoid(z_2)
    
    # 模型预测值与真实值的差，大小为(1, num)
    error = a_2 - y.T
    
    if (j% 1000) == 0:
        print "Error:" + str(np.mean(np.abs(error)))
    
    ## 反向传播，计算损失函数相对于a_1，z2，w_1, a1，z1，w_0的梯度
    ## loss = (1/num)sum(y log a2 + (1 - y) log (1 - a_2))
    
    ## 计算损失函数相对于a_2的梯度
    a_2_delta = np.multiply(error, 1/sigmoid(a_2, derivative = True))
    

    ## 计算损失函数相对于z_2的梯度
    ## z_2_delta = np.multiply((d loss / d a_2), (d a_2 / d z_2))        
    z_2_delta = error
    
    
    ## 计算损失函数相对于w_1的梯度
    ## w_1_delta = (1 / num)np.dot((z_2_delta), (a_1)^T)
    w_1_delta = np.dot(z_2_delta, a_1.T)/float(num)
    
        
    ## 计算损失函数相对于a_1的梯度
    ## a_1_delta = sigmoid_derivative(a_1)
    a_1_delta = np.dot(w_1.T, z_2_delta)
    
    ## 计算损失函数相对于z_1的梯度
    ## z_1_delta = np.multiply((d loss)/(d z_2) (d z_2)/(d a_1),(d a_1)/(d z_1))
    z_1_delta = np.multiply(a_1_delta, sigmoid(a_1, derivative = True))
    
    ## 计算损失函数相对于w_0的梯度
    ## w_0_delta = (1 / num) (d loss)/(d z_1) (d z_1)/(d_w_0)
    w_0_delta = np.dot(z_1_delta, X.T)/float(num)
    
    
    ## 权重更新
    w_0 = w_0 - w_0_delta
    w_1 = w_1 - w_1_delta

## 打印结果
print(w_0)
print('')
print(w_1)