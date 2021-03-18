import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime


class Config:
    batch_size = 64
    epoch = 2
    alpha = 1e-3
    print_per_step = 100  # 控制输出


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  
            nn.ReLU()
        )

        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.output = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (_, _) = self.lstm(x, None)

        out = self.output(r_out[:, -1, :])
        return out


class TrainProcess:

    def __init__(self, model="CNN"):
        self.train, self.test = self.load_data()
        self.model = model
        if self.model == "CNN":
            self.net = CNN()
        elif self.model == "LSTM":
            self.net = LSTM()
        else:
            raise ValueError('"CNN" or "LSTM" is expected, but received "%s".' % model)
        self.criterion = nn.CrossEntropyLoss()  # 定义损失函数
        self.optimizer = optim.Adam(self.net.parameters(), lr=Config.alpha)

    @staticmethod
    def load_data():
        print("Loading Data......")

        train_data = datasets.MNIST(root='./data/',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

        test_data = datasets.MNIST(root='./data/',
                                   train=False,
                                   transform=transforms.ToTensor())

        # 返回一个数据迭代器
        # shuffle：是否打乱顺序
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=Config.batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=Config.batch_size,
                                                  shuffle=False)
        return train_loader, test_loader

    def train_step(self):
        print("Training & Evaluating based on '%s'......" % self.model)
        file = './result/raw_train_mnist.txt'
        fp = open(file,'w',encoding='utf-8')
        fp.write('epoch\tbatch\tloss\taccuracy\n')
        for epoch in range(Config.epoch):
            print("Epoch {:3}.".format(epoch + 1))

            for batch_idx,(data, label) in enumerate(self.train):
                data, label = Variable(data.cpu()), Variable(label.cpu())
                # LSTM输入为3维，CNN输入为4维
                if self.model == "LSTM":
                    data = data.view(-1, 28, 28)
                self.optimizer.zero_grad()  
                outputs = self.net(data)  
                loss = self.criterion(outputs, label)  
                loss.backward() 
                self.optimizer.step() 

                # 每100次打印一次结果
                if batch_idx % Config.print_per_step == 0:
                    _, predicted = torch.max(outputs, 1)
                    correct = int(sum(predicted == label)) 
                    accuracy = correct / Config.batch_size  
                    msg = "Batch: {:5}, Loss: {:6.2f}, Accuracy: {:8.2%}."
                    print(msg.format(batch_idx, loss, accuracy))
                    fp.write('{}\t{}\t{}\t{}\n'.format(epoch,batch_idx,loss,accuracy))
        fp.close()
        test_loss = 0.
        test_correct = 0
        for data, label in self.test:
            data, label = Variable(data.cpu()), Variable(label.cpu())
            if self.model == "LSTM":
                data = data.view(-1, 28, 28)
            outputs = self.net(data)
            loss = self.criterion(outputs, label)
            test_loss += loss * Config.batch_size
            _, predicted = torch.max(outputs, 1)
            correct = int(sum(predicted == label))
            test_correct += correct

        accuracy = test_correct / len(self.test.dataset)
        loss = test_loss / len(self.test.dataset)
        print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))
        torch.save(self.net.state_dict(),'./result/raw_train_mnist_model.pth')


if __name__ == "__main__":
    p = TrainProcess(model='CNN')
    p.train_step()