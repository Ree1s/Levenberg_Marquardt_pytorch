import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime
import functools
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import datasets, transforms
from torch.nn import init



class Config:
    batch_size = 64
    epoch = 2
    alpha = 1e-3
    print_per_step = 100  # 控制输出

 
 

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
 



class DampingAlgorithm(object):
    def __init__(self,
                 starting_value = 1e-3,
                 dec_factor = 0.1,
                 inc_factor = 10.0,
                 min_value = 1e-10,
                 max_value = 1e+10,
                 fletcher = False):
        self.starting_value = starting_value
        self.dec_factor = dec_factor
        self.inc_factor = inc_factor
        self.min_value = min_value
        self.max_value = max_value
        self.fletcher = fletcher     

    def init_step(self, damping_factor, loss):
        return damping_factor

    def decrease(self, damping_factor, loss):
        return torch.max(damping_factor*self.inc_factor,self.min_value)

    def increase(self,damping_factor,loss):
        return torch.min(damping_factor*self.inc_factor,self.max_value)

    def stop_training(self,damping_factor,loss):
        return damping_factor >= self.max_value

    def apply(self,damping_factor,JJ):
        if self.fletcher:
            damping = JJ.diag().diag()
        else:
            damping = torch.eye(JJ.shape[0],dtype=JJ.dtype)
        damping = torch.mul(damping_factor,damping)
        return torch.add(JJ,damping)




def raw_jacobian(inputs, outputs):
    jac = [torch.autograd.grad(inputs, i,allow_unused=True,retain_graph=True) for i in outputs]
    return jac


def compute_jacobian(model, loss, inputs, targets):
    outputs = model(inputs)
    residuals = loss(outputs,targets)
    jacobians = raw_jacobian(residuals,model.parameters())
    num_residuals = torch.prod(torch.tensor(residuals.shape))
    reshape_size = int(num_residuals.detach().numpy().item())
    new_jacobians = []
    for j in jacobians:
        for i in j:
            new_jacobians.append(i.reshape(reshape_size,-1))
    jacobian = torch.cat(new_jacobians, dim=1)
    residuals = torch.reshape(residuals, (reshape_size, -1))
    return jacobian, residuals, outputs


def init_gauss_newton_overdetermined(model, loss, inputs, targets):
    slice_size = jacobian_max_num_rows // _num_outputs
    batch_size = inputs.shape[0]
    num_slices = batch_size // slice_size
    remainder = batch_size % slice_size

    JJ = torch.zeros([_num_variables, _num_variables])

    rhs = torch.zeros([_num_variables, 1])

    outputs_array = []
    for i in torch.arange(num_slices):

        _inputs = inputs[i * slice_size:(i + 1) * slice_size]
        _targets = targets[i * slice_size:(i + 1) * slice_size]
        J,residuals,_outputs = compute_jacobian(model, loss, inputs, targets)
        outputs_array.append(_outputs)
        JJ += torch.matmul(J.T,J)
        rhs +=torch.matmul(J.T,residuals)

    if remainder > 0:
        _inputs = inputs[num_slices * slice_size::]
        _targets = targets[num_slices * slice_size::]

        J, residuals, _outputs = compute_jacobian(model, loss, inputs, targets)

        if num_slices > 0:
            outputs = torch.cat([torch.stack(outputs_array).flatten(),_outputs],dim=0)
        else:
            outputs = _outputs
        JJ += torch.matmul(J.T,J)
        rhs += torch.matmul(J.T, residuals)

    else:
        outputs = torch.stack(outputs_array).flatten()           

    return 0.0, JJ, rhs, outputs


def init_gauss_newton_underdetermined(model, loss, inputs, targets):
    J, residuals, outputs = compute_jacobian(model, loss, inputs, targets)
    JJ = torch.matmul(J, J.T)
    rhs = residuals
    return J, JJ, rhs, outputs

def compute_gauss_newton_overdetermined(J, JJ, rhs):
    updates = solve_function(JJ, rhs)
    return updates

def compute_gauss_newton_underdetermined(J, JJ, rhs):
    updates = solve_function(JJ, rhs)
    updates = torch.matmul(J.T, updates)
    return updates




def compute_num_outputs(model, loss,inputs, targets):
    outputs = model(inputs)
    residuals = loss.residuals(targets,outputs)
    return torch.prod(torch.tensor(residuals.shape[1::]))


# Define and select linear system equation solver.
def qr(matrix, rhs):
    q, r = torch.qr(matrix)
    y = torch.matmul(q.T, rhs)
    return torch.triangular_solve(r, y).solution


def cholesky(matrix, rhs):
    chol = torch.cholesky(matrix)
    return torch.cholesky_solve(chol, rhs)


def solve(matrix, rhs):
    return torch.solve(matrix, rhs)





batch_size = 64
_num_outputs=0
jacobian_max_num_rows = 100
attempts_per_step = 10
damping_algorithm = DampingAlgorithm()
solve_function = qr  ## or cholesky  slove会出现0情况
alpha=1e-3


#创建fake data
# torch.manual_seed(99)
# x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
# y = x.pow(2) + 0.1 * torch.normal(torch.zeros(x.size()))
# plt.scatter(x.numpy(), y.numpy())
# plt.show()
 
np.random.seed(666)
X = np.linspace(-1, 1, 1000)
y = np.power(X, 2) + 0.1 * np.random.normal(0, 1, X.size)
print(X.shape)
print(y.shape)
# plt.scatter(X, y)
# plt.show()
 
# 创建训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1024)
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_train = torch.unsqueeze(X_train, dim=1)  #转换成二维
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_train = torch.unsqueeze(y_train, dim=1)
print(X_train.type)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
X_test = torch.unsqueeze(X_test, dim=1)  #转换成二维
 
# train_size = int(0.7 * len(X))
# test_size = len(X) - train_size
# X_train, X_test = Data.random_split(X, [train_size, test_size])
# print(len(X_train), len(X_test))

 
#将数据装载镜data中, 对数据进行分批训练
torch_data  = Data.TensorDataset(X_train, y_train)
loader = Data.DataLoader(dataset=torch_data, batch_size=batch_size, shuffle=True)
model = Net()
criterion = nn.MSELoss()



_backup_variables = []
_splits = []
_shapes = []
        
for variable in model.parameters():
    variable_shape = variable.shape
    variable_size = torch.prod(torch.tensor(variable_shape))
    backup_variable = torch.autograd.Variable(
        torch.zeros_like(variable),
        requires_grad=False
    )
    _backup_variables.append(backup_variable)
    _splits.append(variable_size)
    _shapes.append(variable_shape)

_num_variables = torch.sum(torch.tensor(_splits)).numpy().item()




# model.train()
for epoch in range(1):
    for batch_idx,(data, label) in enumerate(loader):
        model.train()
        

        if not _num_outputs:
            J, JJ, rhs, outputs = init_gauss_newton_underdetermined(model,criterion,data,label)
        else:
            ### 尽量不使用，此部分会内存爆炸
            J, JJ, rhs, outputs = init_gauss_newton_overdetermined(model,criterion,data,label)
        
        normalization_factor = 1.0 / torch.tensor(batch_size)
        JJ *= normalization_factor
        rhs *= normalization_factor
        loss = criterion(outputs,label)
        stop_training = False
        attempt = 0
        
        damping_factor = torch.autograd.Variable(
            torch.tensor(
                damping_algorithm.starting_value
            ),
            requires_grad = False
        )
        
        damping_factor = damping_algorithm.init_step(damping_factor,loss)
        attempts = torch.tensor(attempts_per_step, dtype = torch.int32)
        JJ_damped = damping_algorithm.apply(damping_factor, JJ)
        updates = compute_gauss_newton_underdetermined(J, JJ_damped, rhs)
        updates = torch.split(torch.squeeze(updates,dim=-1), _splits)
        updates = [torch.reshape(update, shape) for update, shape in zip(updates, _shapes)]
#         print(updates[0])
#         print([-1*x for x in updates][0])
        
        prev_loss=loss.item()
        model.eval()
        cnt=0
        model.zero_grad()

        for idx,p in enumerate(model.parameters()):
            # p.requires_grad_(False)
            p.detach_()
            # print('1: ',p[0])
            p+=updates[idx].reshape(p.shape)
#             p.requires_grad=False
#             p+=updates[idx].reshape(p.shape)
#             cnt+=num
            p.requires_grad_(True)
            # print('2: ',p[0])

        outputs = model(data)
        loss_out = criterion(outputs,label)
        if batch_idx % 100 == 0:
            print('{}:{}'.format(batch_idx,loss_out.detach().numpy().item()))
        if loss_out.detach().numpy().item() < prev_loss:
            alpha/=10
        else:
            alpha*=10
            cnt=0
            for idx,p in enumerate(model.parameters()):
                # p.requires_grad_(False)
                p.detach_()
                # print('1: ',p[0])
                p+=updates[idx].reshape(p.shape)
    #             p.requires_grad=False
    #             p+=updates[idx].reshape(p.shape)
    #             cnt+=num
                p.requires_grad_(True)
                # print('2: ',p[0])
            
        # print('ok')
#         break


