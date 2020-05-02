import pickle
import numpy as np
import matplotlib.pyplot as plt
import time


class SVMClassifier:
    def __init__(self):
        self.W = None

    def loss(self, X, y, reg):
        numOfTrain = X.shape[0]
        scores = X.dot(self.W)
        corClassScores = scores[range(numOfTrain), list(y)].reshape(-1, 1)

        # s(j) - s(yi) and take it compared with 0, take max
        tmps = np.maximum(0, scores - corClassScores + 1)
        # remove the val when j == yi
        tmps[range(numOfTrain), list(y)] = 0
        # compute loss based on hinge loss and regularization
        losses = np.sum(tmps) / numOfTrain + reg * np.sum(self.W * self.W)

        numOfClasses = self.W.shape[1]
        mat = np.zeros((numOfTrain, numOfClasses))
        mat[tmps > 0] = 1
        # s <= 0
        mat[range(numOfTrain), list(y)] = 0
        # j = yi & s > 0
        mat[range(numOfTrain), list(y)] = -np.sum(mat, axis=1)

        dW = X.T.dot(mat)
        dW /= numOfTrain
        dW += 2 * reg * self.W

        return losses, dW

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, isRateOfProgressShown=False):
        numOfTrain, dimensions = X.shape
        numOfClasses = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dimensions, numOfClasses)

        loss_history = []
        print("Begin svm iterations!")
        for it in range(num_iters):
            # randomly take samples
            idx_batch = np.random.choice(numOfTrain, batch_size, replace=True)
            X_batch = X[idx_batch]
            Y_batch = y[idx_batch]

            loss, grad = self.loss(X_batch, Y_batch, reg)
            loss_history.append(loss)

            # gradient descent
            self.W -= learning_rate * grad

            if isRateOfProgressShown and (it + 1) % 500 == 0:
                print("iteration %d / %d: loss %f" % (it+1, num_iters, loss))

        return loss_history

    def predict(self, X):
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred


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

        X = imgTmp['data']
        Y = imgTmp['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        trainData.append(X)
        trainLabel.append(Y)

    # get the test data
    fileName = filePath + "test_batch"
    fo = open(fileName, 'rb')
    imgTmp = pickle.load(fo, encoding="latin1")
    X = imgTmp['data']
    Y = imgTmp['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    testData.append(X)
    testLabel.append(Y)

    return np.concatenate(trainData), np.concatenate(trainLabel), np.concatenate(testData), np.concatenate(testLabel)


def main():
    print("data preprocessing!")
    # get the initial image data
    X_train, y_train, X_test, y_test = fileLoads("./cifar-10-batches-py/")
    '''print('Training data with RGB channels shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data with RGB channels shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)'''

    # convert RGB channels to single channel
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    '''print('Training data with single channel shape: ', X_train.shape)
    print('Test data with single channel shape: ', X_test.shape)'''

    # preprocessed: compute the image mean based on the training image data, and the train/test data subtracts it
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    # add one row(1s) to the end of the X_data in order tp compute conveniently
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    # print(X_train.shape, X_test.shape)

    # SVM train
    svm = SVMClassifier()
    time_start = time.time()
    # learning_rate=1e-8  |  reg=2.5e4  |  num_iters=50000
    #loss_history = svm.train(X_train, y_train, learning_rate=1e-8, reg=2.5e4, num_iters=50000, isRateOfProgressShown=True)
    # learning_rate=1e-7 |  reg=2.5e4  |  num_iters=3000
    # loss_history = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4, num_iters=3000, isRateOfProgressShown=True)
    # learning_rate=1e-7 |  reg=2.5e3  |  num_iters=10000
    loss_history = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e3, num_iters=10000, isRateOfProgressShown=True)
    time_finish = time.time()
    print('Total processing time is %fs' % (time_finish - time_start))
    plt.plot(loss_history)
    plt.xlabel('Iteration times')
    plt.ylabel('Loss value')
    plt.show()

    y_train_pre = svm.predict(X_train)
    print('training data set accuracy: %f' % (np.mean(y_train == y_train_pre),))

    y_test_pre = svm.predict(X_test)
    print('testing data set accuracy: %f' % (np.mean(y_test == y_test_pre),))


if __name__ == '__main__':
    main()
