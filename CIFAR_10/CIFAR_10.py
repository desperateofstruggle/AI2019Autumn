import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Datas
import torch.optim as optim
from torch.autograd import Variable

from torchvision import transforms

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            # (3*32*32) -> (6*28*28) -> (6*14*14) Convolutional layer
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            # set the output channel
            nn.BatchNorm2d(6),
            # decrease the memory usage
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            # (6*14*14) -> (16*10*10) -> (16*5*5) Convolutional layer
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            # set the output channel
            nn.BatchNorm2d(16),
            # decrease the memory usage
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.outlayer = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 16*5*5)
        x = self.outlayer(x)
        return x


def main():
    IS_NEED_DOWNLOAD_MNIST = True
    BATCH_SIZE = 4
    LEARNING_RATE = 0.01

    # CIFAR-10
    train_data = torchvision.datasets.CIFAR10(
        root='./data/',
        train=True,
        # transform=torchvision.transforms.ToTensor(),
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
        download=IS_NEED_DOWNLOAD_MNIST,
    )

    test_data = torchvision.datasets.CIFAR10(
        root='./data/',
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
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
    lossFun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # train
    epoch = 0
    for epoch in range(10):
        running_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
            imgData = x
            labelData = y
            imgData = Variable(imgData)
            labelData = Variable(labelData)

            # forward
            out = model(imgData)
            loss = lossFun(out, labelData)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # epoch += 1
            running_loss += loss.data.item()
            if step % 4000 == 3999:
                print('Training epoch: {}, step: {}, with loss: {:.6}'.format(epoch+1, step+1, running_loss / 4000))
                running_loss = 0.0

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
