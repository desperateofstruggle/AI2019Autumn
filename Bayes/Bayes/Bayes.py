import numpy as np
import os

"""
函数说明:读取词汇数据
Parameters:
    filename - 目标文件的地址及文件名
Return:
    dataArray - 返回int型数据列表(元素为[文本号 词号 出现次数])下标从0起
"""


def fileToNumpy(filename):
    file = open(filename)
    file_lines = file.readlines()
    numberOfLines = len(file_lines)
    dataArray = np.zeros((numberOfLines, 3))
    index = 0
    for line in file_lines:
        line = line.strip()
        formLine = line.split(' ')
        dataArray[index, :] = formLine[0:3]
        index += 1
    return dataArray.astype(int)


"""
函数说明:读取文本是否为垃圾邮件数据
Parameters:
    filename - 目标文件的地址及文件名
Return:
    dataArray - 返回int型数据列表(元素为0——垃圾邮件或1——正常邮件)下标从1起，对应文本号
"""


def fileToNumpy_label(filename):
    file = open(filename)
    file_lines = file.readlines()
    numberOfLines = len(file_lines)
    dataArray = np.zeros((numberOfLines + 1, 1))
    index = 1
    for line in file_lines:
        line = line.strip()
        formLine = line.split(' ')
        dataArray[index] = formLine[0]
        index = index + 1
    return dataArray.astype(int)


"""
函数说明:朴素贝叶斯分类器训练函数
Parameters:
    trainMatrix - 训练文本矩阵(即每篇文本对应所有词号是否存在的矩阵)
    trainCategory - 训练类别标签向量(即对应所有文本是否为垃圾邮件的01矩阵)
Return:
    p0Vect - 正常邮件类的条件概率数组
    p1Vect - 垃圾邮件类的条件概率数组
    pSpam  - 文本属于垃圾邮件类的概率
"""


def trainBayes(trainMatrix, trainCategory):
    numOfTrainText = len(trainMatrix)  # 训练文本数目
    numOfWords = len(trainMatrix[0])  # 记录词库总数目
    pSpam = sum(trainCategory) / float(numOfTrainText)  # 计算 P(文本属于垃圾邮件类)
    p0NumOfnonspam = np.ones(numOfWords)  # 词条出现数初始化为1,为拉普拉斯平滑(对应正常邮件)
    p1NumOfspam = np.ones(numOfWords)  # 词条出现数初始化为1,为拉普拉斯平滑(对应垃圾邮件)
    p0Denom = 2.0  # 分母初始化为2 ,为拉普拉斯平滑
    p1Denom = 2.0
    for i in range(0, numOfTrainText):
        if trainCategory[i] == 1:  # 计算垃圾邮件的条件概率相关数据，即P(xk|1)···k=1,2,3....
            p1NumOfspam += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 计算正常邮件的条件概率相关数据，即P(xk|0)···k=1,2,3....
            p0NumOfnonspam += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1NumOfspam / p1Denom)
    p0Vect = np.log(p0NumOfnonspam / p0Denom)  # 取对数，防止下溢出
    return p0Vect, p1Vect, pSpam


"""
函数说明:朴素贝叶斯分类器分类函数
Parameters:
	matrix - 待判断的文本数组
	p0Vec - 正常邮件的条件概率数组
	p1Vec - 垃圾邮件的条件概率数组
	pClass1 - 文本属于垃圾邮件的概率
Return:
	0 - 正常邮件
	1 - 垃圾邮件
"""


def checks(matrix, p0Vec, p1Vec, pClass1):
    p1 = sum(matrix * p1Vec) + np.log(pClass1)
    p0 = sum(matrix * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


"""
函数说明:整体功能逻辑实现函数
"""


def main():
    filename = "./data/train-features.txt"  # 训练数据文件的相对地址
    dataArray_train_features = fileToNumpy(filename)

    sizes = dataArray_train_features.astype(int)[-1][0]
    lens = len(dataArray_train_features)
    trainMat = []
    trainClasses = []

    cnt = 0
    for i in range(1, sizes + 1):
        vocabList = list(range(2501))  # 共存在这么多的词汇
        vocabList = [0] * len(vocabList)
        while cnt < lens and dataArray_train_features.astype(int)[cnt][0] == i:
            vocabList[dataArray_train_features.astype(int)[cnt][1]] = 1
            cnt = cnt + 1
        trainMat.append(vocabList)

    filename = "./data/train-labels.txt"  # 训练文本是否为垃圾邮件的文件的相对地址
    dataArray_train_labels = fileToNumpy_label(filename)
    label_sizes = len(dataArray_train_labels)

    for i in range(1, label_sizes):
        trainClasses.append(dataArray_train_labels[i])

    p0, p1, pSpam = trainBayes(np.array(trainMat), np.array(trainClasses))  # 训练朴素贝叶斯模型

    # 获取test测试数据
    filename = "./data/test-features.txt"  # 测试数据文件的相对地址
    dataArray_test_features = fileToNumpy(filename)
    sizes = dataArray_test_features.astype(int)[-1][0]
    lens = len(dataArray_test_features)

    testSet = []

    cnt = 0
    for i in range(1, sizes + 1):
        vocabList = list(range(2501))  # 共存在这么多的词汇
        vocabList = [0] * len(vocabList)
        while cnt < lens and dataArray_test_features.astype(int)[cnt][0] == i:
            vocabList[dataArray_test_features.astype(int)[cnt][1]] = 1
            cnt = cnt + 1
        testSet.append(vocabList)

    # 获取测试标准结果
    filename = "./data/test-labels.txt"  # 训练数据是否为垃圾邮件的文件的相对地址
    savefile = "./result/result.txt"
    dataArray_test_labels = fileToNumpy_label(filename)

    errorCount = 0
    cnt = 1
    for tests in testSet:
        if checks(np.array(tests), p0, p1, pSpam) != dataArray_test_labels[cnt][0]:  # 如果判断错误
            errorCount += 1
            print("判断错误的测试集文件号：%d" % cnt)
        with open(savefile, 'a+') as f:
            f.write(str(checks(np.array(tests), p0, p1, pSpam)) + "\n")
        cnt = cnt + 1
    print('误判率：%f%%' % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    main()
