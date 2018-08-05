import numpy as np

f = open("./Data/train_data.txt")
train_data = []
for line in f:
    a = line.strip().split(" ")
    train_data.append(a)
f.close()
f = open("./Data/train_label.txt")
train_label = []
for line in f:
    a = int(line.strip().split(" ")[0])
    train_label.append(a)
f.close()
f = open("./Data/vocab_list.txt")
vocab = []
for line in f:
    a = line.strip().split(" ")[0]
    vocab.append(a)
f.close()
f = open("./Data/test_data.txt")
test_data = []
for line in f:
    a = line.strip().split(" ")
    test_data.append(a)
f.close()
f = open("./Data/test_label.txt")
test_label = []
for line in f:
    a = int(line.strip().split(" ")[0])
    test_label.append(a)
f.close()

def createVector(vocab, mail_text):
    '''
    输入词库列表vocab，和单个分词后的邮件内容列表mail_text，
    输出参数为向量化文本列表mail_vec
    '''

    ## 获取vocab的长度
    len_vocab = len(vocab)

    ## 零值初始化向量，长度为len_vocab
    mail_vec = [0 for i in range(len_vocab)]

    ## 遍历mail_text中的元素
    for value in mail_text:

        ## 判断value在vocab中的位置，将mail_vec对应位置上的元素值加1
        if value in vocab:

            ## 获取value在vaocab中的索引value_index
            value_index = vocab.index(value)

            ## mail_vec中value_index位置上的元素赋值为1
            mail_vec[value_index] = 1
        else:
            continue

    ## 返回结果
    return mail_vec

## 使用列表推导式将train_data向量化，结果保存在train_matrix
train_matrix = [createVector(vocab, text) for text in train_data]
test_matrix = [createVector(vocab, text) for text in test_data]
def createNBClassifier(train_matrix, train_label):
    
    import numpy as np
    
    ## 获取训练集的数量num
    num = len(train_matrix)
    ## 获取train_matrix的维度num_col
    num_col = len(train_matrix[0])
    
    ## 统计c = 1下x_{i}的出现的频次列表，形如[num(x_1|c = 1), num(x_2|c=1),...];同理 c = 0
    ## 为了防止出现概率为零的情况，我们将所有词出现的频次初始化为1
    ## 使用np.ones()方法p1_num和p0_num的类型为ndarray
    p1_num = np.ones(num_col)
    p0_num = np.ones(num_col)
    
    ## 垃圾邮件和正常邮件出现的单词的数量  
    p1_vec_denom = 0.
    p0_vec_denom = 0.
    
    ## 统计频次
    for index in range(num):
        
        ## 垃圾邮件
        if train_label[index] == 1:
            ## 单词出现的位置值为1，不出现值为0，因此可以直接相加，累计词频，train_matrix的元素为list
            
            ## 需要转换为ndarray，便于向量直接相加
            p1_num = p1_num + np.array(train_matrix[index])
            ## 统计垃圾邮件中的单词数
            p1_vec_denom = p1_vec_denom + np.sum(train_matrix[index])
        
        ## 正常邮件
        else:
            ## 单词出现的位置值为1，不出现值为0，因此可以直接相加，累计词频，train_matrix的元素为list
            ## 需要转换为ndarray，便于向量直接相加
            p0_num = p0_num + np.array(train_matrix[index])
            ## 统计正常邮件中的单词数
            p0_vec_denom = p0_vec_denom + np.sum(train_matrix[index])
    
    ####根据从训练集统计的频次
    ##计算[p(x_i | c = 1), p(x_2|c = 1)...] 以及 [p(x_i) | c = 0), p(x_i|c=0),...]
    ####为了防止向下溢出，概率取对数;为了配合初始化值为1的频次，p1_vec的分母取值增加2
    p1_vec = np.log(p1_num/(p1_vec_denom + 2))
    p0_vec = np.log(p0_num/(p0_vec_denom + 2))
    
    ## 计算垃圾邮件出现的概率
    pc1 = np.sum(train_label)/float(num)
    
    ## 计算正常邮件出现的概率
    pc0 = 1 - pc1
    
    ##返回结果
    return p1_vec, p0_vec, pc1, pc0


## 使用自定义函数返回结果p1_vec, p0_vec, pc1, pc0
p1_vec, p0_vec, pc1, pc0 = createNBClassifier(train_matrix, train_label)
#print(p1_vec, p0_vec, pc1, pc0)


def predict(test_matrix, p1_vec, p0_vec, pc1, pc0):
    
   
    ## 声明预测标签列表，初始值为0
    pred_label = [0 for i in range(len(test_matrix))]
    
    ## 遍历测试机中的每一个样本
    for index, record in enumerate(test_matrix):
        ## 将list转换为ndarray
        record_array = np.array(record)
        
        ## 根据公式 log(p(c|x))正比于 log(p(x_1|c)p(x_2|c)...p(x_k|c)p(c))
        ## 计算测试样本属于垃圾邮件的概率
        p1 = np.dot(record_array, p1_vec) + np.log(pc1)
    
        ## 计算测试样本属于正常邮件的概率
        p0 = np.dot(record_array, p0_vec) + np.log(pc0)
        
        ## 确认预测标签
        if p1 > p0:
            ##将索引为index的测试标签值改为1
            pred_label[index] = 1
        else:
            continue
    
    ## 返回结果
    return pred_label

## 使用测试数据集预测标签
pred_label = predict(test_matrix, p1_vec, p0_vec, pc1, pc0)

## 获取预测标签与真实标签的差，diff类型为ndarray
diff = np.array(pred_label) - np.array(test_label)

## 计算错误率
error_rate = sum(diff)/float(len(diff))

## 打印错误率
print(error_rate)


from sklearn.naive_bayes import GaussianNB
X = np.array(train_matrix)
#X.reshape(-1,1)
y = np.array(train_label)
clf = GaussianNB()
clf.fit(X,y)

pred_label_1 = clf.predict(np.array(test_matrix))
diff_1 = np.array(pred_label_1) - np.array(test_label)
error_rate_1 = sum(diff_1)/float(len(diff))
print(error_rate_1)