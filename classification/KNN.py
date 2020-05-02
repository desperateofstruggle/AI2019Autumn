import pickle
import numpy as np


'''
function declaration: the file read function
Parameters:
    filePath    -   the file path ("xxx/", example -> "./data/")
Returns:
    trainData   -   the train set of data (np.array()  [[],[],...,[]])
    trainLabel  -   the train set of labels (np.array()  [])
    testData    -   the test set of data (np.array()  [[],[],...,[]])
    testLabel   -   the test set of labels (np.array()  [])
'''

def fileLoads(filePath):
    files = 'data_batch_'
    trainData = []
    trainLabel = []
    testData = []
    testLabel = []

    # get the train data
    for i in range(1, 6):
        fileName = filePath + files + str(i)
        fo = open(fileName, 'rb')
        imgTmp = pickle.load(fo, encoding="latin1")
        trainData.extend(list(imgTmp["data"])[0:])
        trainLabel.extend(list(imgTmp["labels"])[0:])

    # get the test data
    fileName = filePath + "test_batch"
    fo = open(fileName, 'rb')
    imgTmp = pickle.load(fo, encoding="latin1")
    testData.extend(list(imgTmp["data"]))
    testLabel.extend(list(imgTmp["labels"]))
    return np.array(trainData), np.array(trainLabel), np.array(testData), np.array(testLabel)


'''
function declaration: the main process of knn
Parameters:
    XTrain  - the train set of data (np.array()  [[],[],...,[]])
    YTrain  - the train set of labels (np.array()  [])
    XTest   - the test set of data (np.array()  [[],[],...,[]])
    k       - the k parameter

Returns:
    predictResult   -   the prediction result (np.array()  [])
'''

def knnMain(XTrain, YTrain, XTest, k=20):
    lens = XTest.shape[0]
    predictResult = []
    for i in range(lens):
        if (i+1) % 5000 == 0:
            print("Process on the", i + 1, "th images")
        distances = np.sum(np.square(XTrain - XTest[i]), axis=1)
        distancesSorted = np.argsort(distances)[0:k]

        classSort = YTrain[distancesSorted]
        classCount = np.bincount(classSort)
        predi = np.argmax(classCount)
        predictResult.append(predi)

    return np.array(predictResult)


def main():
    k = input("Please input the k to be applied(-1 to scope automatically):")
    XTrain, YTrain, XTest, YTest = fileLoads("./cifar-10-batches-py/")
    if k == "-1":
        KS = [1, 3, 5, 10, 20, 50, 100]
        accus = []
        for k in KS:
            print("Proccess on k ==", k)
            predictResult = knnMain(XTrain, YTrain, XTest, int(k))
            accu = np.mean(predictResult == YTest)
            accus.append(accu)
        for j in range(len(accus)):
            print("The accuracy of the prediction is ", accus[j] * 100, "% with k ==", KS[j])
    else:
        print("The len of test set is ", len(XTest))
        predictResult = knnMain(XTrain, YTrain, XTest, int(k))
        print("The predicted results are: ", predictResult)
        print("The Test Labels are: ", YTest)

        accu = np.mean(predictResult == YTest)
        print("The accuracy of the prediction is ", accu * 100, "%")


if __name__ == '__main__':
    main()
