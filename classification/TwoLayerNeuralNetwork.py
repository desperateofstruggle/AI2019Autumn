import pickle
import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNeturalNetwork:
    def __init__(self):
        self.W2 = None
        self.W1 = None

    def loss(self, X, y, reg=0.0):
        W1 = self.W1
        W2 = self.W2

        # get the num of train data and len of each data
        numOfTrain, dimension = X.shape

        # compute the forward pass
        hiddenLayerScores = np.maximum(0, np.dot(X, W1))
        foutScores = np.dot(hiddenLayerScores, W2)

        # compute the Delta s between true goal class and other class
        foutScores -= foutScores.max(axis=1).reshape(numOfTrain, 1)

        # compute the loss
        loss = np.log(np.exp(foutScores).sum(axis=1)).sum() - foutScores[range(numOfTrain), y].sum()
        loss /= numOfTrain + reg * (np.sum(W2 * W2) + np.sum(W1 * W1))

        # compute the gradients
        grads = {}
        dW = np.exp(foutScores) / (np.exp(foutScores).sum(axis=1)).reshape(numOfTrain, 1)
        dW[range(numOfTrain), y] -= 1
        dW /= numOfTrain

        grads["W2"] = np.dot(hiddenLayerScores.T, dW) + reg * W2

        dWHidden = np.dot(dW, W2.T)
        dWHidden[hiddenLayerScores < 0.00001] = 0

        grads["W1"] = np.dot(X.T, dWHidden) + reg * W1

        return loss, grads

    def train(self, X, y, hidden_size=100, learning_rate=1e-3, learning_rate_decay=0.95, reg=1e-5, num_iters=100, batch_size=200, isRateOfProgressShown=False):
        numOfTrain, dimensions = X.shape
        numOfClasses = np.max(y) + 1
        iterations_per_epoch = max(numOfTrain / batch_size, 1)

        if self.W1 is None or self.W2 is None:
            self.W1 = 1e-4 * np.random.randn(dimensions, hidden_size)
            self.W2 = 1e-4 * np.random.randn(hidden_size, numOfClasses)

        loss_history = []
        print("Begin iterations!")
        for it in range(num_iters):
            # randomly take samples
            idx_batch = np.random.choice(numOfTrain, batch_size, replace=True)
            X_batch = X[idx_batch]
            Y_batch = y[idx_batch]

            loss, grads = self.loss(X_batch, Y_batch, reg)
            loss_history.append(loss)

            # gradient descent
            self.W1 -= learning_rate * grads["W1"]
            self.W2 -= learning_rate * grads["W2"]

            if isRateOfProgressShown and (it + 1) % 500 == 0:
                print("iteration %d / %d: loss %f" % (it+1, num_iters, loss))

            if it % iterations_per_epoch == 0:
                # Decay learning rate
                learning_rate *= learning_rate_decay

        return loss_history

    def predict(self, X):
        W1 = self.W1
        W2 = self.W2

        hidden = np.maximum(0, X.dot(W1))
        scores = hidden.dot(W2)
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

    neu = TwoLayerNeturalNetwork()
    #loss_history = neu.train(X_train, y_train, hidden_size=500, num_iters=20000, isRateOfProgressShown=True)
    loss_history = neu.train(X_train, y_train, hidden_size=500, num_iters=5000, batch_size=500, reg=0.001, isRateOfProgressShown=True)

    plt.plot(loss_history)
    plt.xlabel('Iteration times')
    plt.ylabel('Loss value')
    plt.show()

    y_train_pre = neu.predict(X_train)
    print('training data set accuracy: %f' % (np.mean(y_train == y_train_pre),))

    y_test_pre = neu.predict(X_test)
    print('testing data set accuracy: %f' % (np.mean(y_test == y_test_pre),))

    # crossing-validation
    '''learning_rates = [1e-3, 1e-4, 1e-5]
    batch_size = [200, 400, 600]
    best_acc = 0
    best_lr = 0
    best_bs = 0
    input_size = 32 * 32 * 3
    num_classes = 10

    for lr in learning_rates:
        for bs in batch_size:
            for hidden_size in [50, 100, 200]:
                net = TwoLayerNeturalNetwork()
                vnet = net.train(X_train, y_train, hidden_size=hidden_size,
                                 num_iters=3000, batch_size=bs,
                                 learning_rate=lr, learning_rate_decay=0.95,
                                 reg=0.5, isRateOfProgressShown=True)

                val_acc = (net.predict(X_test) == y_test).mean()

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_lr = lr
                    best_bs = bs
                    best_hidden_size = hidden_size
                    best_net = net

    print(best_acc, best_lr, best_bs, best_hidden_size)'''


if __name__ == '__main__':
    main()
