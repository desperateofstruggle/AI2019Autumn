import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Datas
import torch.optim as optim
from torch.autograd import Variable

from torchvision import transforms

# CNN Class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            # (1*28*28) -> (16*26*26)  Convolutional layer
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            # set the output channel
            nn.BatchNorm2d(16),
            # decrease the memory usage
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            # (16*26*26) -> (32*24*24) -> (32*12*12)  Convolutional layer & Pooling layer
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            # set the output channel
            nn.BatchNorm2d(32),
            # decrease the memory usage
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            # (32*12*12) -> (64*10*10)  Convolutional layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # set the output channel
            nn.BatchNorm2d(64),
            # decrease the memory usage
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            # (64*10*10) -> (128*8*8) -> (128*4*4)  Convolutional layer & Pooling layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            # set the output channel
            nn.BatchNorm2d(128),
            # decrease the memory usage
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.outlayer = nn.Sequential(
            nn.Linear(128*4*4, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


# CNN Class with LeNet-5
class CNNLeNet_5(nn.Module):
    def __init__(self):
        super(CNNLeNet_5, self).__init__()

        self.layer1 = nn.Sequential(
            # (1*28*28) -> (20*24*24) -> (20*12*12) Convolutional layer
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),
            # set the output channel
            nn.BatchNorm2d(20),
            # decrease the memory usage
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            # (20*12*12) -> (50*8*8) -> (50*4*4)  Convolutional layer & Pooling layer
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5),
            # set the output channel
            nn.BatchNorm2d(50),
            # decrease the memory usage
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.outlayer = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


def main():
    IS_NEED_DOWNLOAD_MNIST = True
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01

    # Mnist
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        # transform=torchvision.transforms.ToTensor(),
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]),
        download=IS_NEED_DOWNLOAD_MNIST,
    )

    test_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    )

    train_loader = Datas.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True  # 乱序
    )

    test_loader = Datas.DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=True  # 乱序
    )

    model = CNN()
    choice = input("Please choose the Net structure to use: 1 --- LeNet-5   2 --- UserDefined:")
    if choice == "1":
        model = CNNLeNet_5()
    elif choice == "2":
        model = CNN()
    else:
        print("meaningless input")
        return
    lossFun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # train
    epoch = 0
    for dTmp in train_loader:
        imgData, labelData = dTmp

        imgData = Variable(imgData)
        labelData = Variable(labelData)

        # forward
        out = model(imgData)
        loss = lossFun(out, labelData)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch += 1
        if epoch % 100 == 0:
            print('Training epoch: {}, with loss: {:.6}'.format(epoch, loss.data.item()))

    # test
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for dTmp in test_loader:
        imgData, labelData = dTmp
        out = model(imgData)
        loss = lossFun(out, labelData)
        # count the loss sum
        eval_loss += loss.data.item() * labelData.size(0)
        # return the max of each rows
        _, predictions = torch.max(out, 1)
        # count the correct numbers
        numCor = (predictions == labelData).sum()
        eval_acc += numCor.item()
    print('Test Loss:{:.6f}, Acc:{:.6f}'.format(eval_loss / (len(test_data)), eval_acc / (len(test_data))))


if __name__ == '__main__':
    main()
