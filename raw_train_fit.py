import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import datasets, transforms
from torch.nn import init
 


 
np.random.seed(666)
X = np.linspace(-1, 1, 1000)
y = np.power(X, 2) + 0.1 * np.random.normal(0, 1, X.size)
print(X.shape)
print(y.shape)


# 创建训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1024)
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_train = torch.unsqueeze(X_train, dim=1)  #转换成二维
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_train = torch.unsqueeze(y_train, dim=1)
print(X_train.type)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
X_test = torch.unsqueeze(X_test, dim=1)  #转换成二维
 

BATCH_SIZE = 50
LR = 0.02
EPOCH = 2
 
#将数据装载镜data中, 对数据进行分批训练
torch_data  = Data.TensorDataset(X_train, y_train)
loader = Data.DataLoader(dataset=torch_data, batch_size=BATCH_SIZE, shuffle=True)
#创建自己的nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, 20)
        self.predict = nn.Linear(20, 1)
 
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
def weights_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight.data)
        # init.xavier_normal(m.bias.data)
 
adam_net = Net()
opt_adam = torch.optim.Adam(adam_net.parameters(), lr=LR)
loss_func = nn.MSELoss()
all_loss = {}
fp = open('./result/raw_train_fit.txt','w',encoding='utf-8')
fp.write('epoch\tbatch\tloss\n')
for epoch in range(EPOCH):
    print('epoch: ', epoch)
    for batch_idx, (b_x, b_y) in enumerate(loader):
        pre = adam_net(b_x)
        loss = loss_func(pre, b_y)
        opt_adam.zero_grad()
        loss.backward()
        opt_adam.step()
        # print(loss)
        all_loss[epoch+1] = loss
        print('batch: {}, loss: {}'.format(batch_idx,loss.detach().numpy().item()))
        fp.write('{}\t{}\t{}\n'.format(epoch,batch_idx,loss.detach().numpy().item()))
fp.close()
torch.save(adam_net.state_dict(),'./result/raw_train_fit_model.pth')
#对测试集进行预测
adam_net.eval()
predict = adam_net(X_test)
predict = predict.data.numpy()
plt.scatter(X_test.numpy(), y_test, label='origin')
plt.scatter(X_test.numpy(), predict, color='red', label='predict')
plt.legend()
plt.show()
