import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime
import numpy as np
import functools
import matplotlib.pyplot as plt


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



    ## 初始化系数
    def init_step(self, damping_factor, loss):
        return damping_factor

    # 判定上升
    def decrease(self, damping_factor, loss):
        return torch.max(damping_factor*self.inc_factor,self.min_value)

    # 判定下降
    def increase(self,damping_factor,loss):
        return torch.min(damping_factor*self.inc_factor,self.max_value)

    # 停止训练条件
    def stop_training(self,damping_factor,loss):
        return damping_factor >= self.max_value

    # 
    def apply(self,damping_factor,JJ):
        if self.fletcher:
            damping = JJ.diag().diag()
        else:
            damping = torch.eye(JJ.shape[0],dtype=JJ.dtype)
        damping = torch.mul(damping_factor,damping)
        return torch.add(JJ,damping)




def raw_jacobian(inputs, outputs):
    ### 计算雅克比
    jac = [torch.autograd.grad(inputs, i,allow_unused=True,retain_graph=True) for i in outputs]
    return jac


def compute_jacobian(model, loss, inputs, targets):
    # 通过模型输出计算雅克比
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
    # 超定高斯牛顿系数
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
    # 欠定高斯牛顿系数
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
    # 模型输出和损失
    outputs = model(inputs)
    residuals = loss.residuals(targets,outputs)
    return torch.prod(torch.tensor(residuals.shape[1::]))


# Define and select linear system equation solver.
def qr(matrix, rhs):
    # qr分解
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


train_data = datasets.MNIST(root='./data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_data = datasets.MNIST(root='./data/',
                            train=False,
                            transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=Config.batch_size,
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                            batch_size=Config.batch_size,
                                            shuffle=False)



model = CNN()
criterion = nn.CrossEntropyLoss()

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
    for batch_idx,(data, label) in enumerate(train_loader):
        model.train()
        

        if not _num_outputs:
            J, JJ, rhs, outputs = init_gauss_newton_underdetermined(model,criterion,data,label)
        else:
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
        # 对于模型参数进行赋值更新
        for idx,p in enumerate(model.parameters()):
            with torch.no_grad():
                p-=updates[idx].reshape(p.shape)
                p=p.requires_grad_(True)

        outputs = model(data)
        loss_out = criterion(outputs,label)
        if batch_idx % 100 == 0:
            print('batch: {} is ok.'.format(batch_idx))
            print('{}:{}'.format(batch_idx,loss_out.detach().numpy().item()))
        if loss_out.detach().numpy().item() < prev_loss:
            alpha/=10
        else:
            alpha*=10
            cnt=0
            # 对于模型参数进行赋值更新
            for idx,p in enumerate(model.parameters()):
                with torch.no_grad():
                    p-=updates[idx].reshape(p.shape)
                    p=p.requires_grad_(True)


