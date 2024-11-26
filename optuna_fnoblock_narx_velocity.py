# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator 
from tqdm import tqdm
import optuna
# from joblib import Parallel, delayed

torch.manual_seed(0)
np.random.seed(0)

################################################################
# 2d fourier layers
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1   # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2


        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
 
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y,t), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)       
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
       
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class Block(nn.Module):
    def __init__(self, in_size, out_size, mode1_num, mode2_num):
        super(Block, self).__init__()
        self.filter = SpectralConv2d(in_size, out_size, mode1_num, mode2_num)
        self.ws = nn.Conv2d(in_size, out_size, 1)       
        
    def forward(self, x):
        x1 = self.filter(x)
        x2 = self.ws(x)
        x = x1 + x2
        return x
        

class FNO2d(nn.Module):
    def __init__(self, depth, modes1, modes2, width, fc_dim=128, in_dim=5, out_dim=1):
        """
        input channel is 5 :(IC,BC_q,BC_h,x,y)

        """
        super(FNO2d, self).__init__()
        self.depth = depth
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_dim, self.width)
        self.fc1 = nn.Linear(self.width, fc_dim)
        self.layers1 = Block(self.width, self.width, self.modes1, self.modes2)
        self.layers2 = Block(self.width, self.width, self.modes1, self.modes2)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        
    def forward(self, x, y):
        x = torch.cat((x, y), dim=-1)   
        grid = self.get_grid(x.shape, x.device)       
        x = torch.cat((x, grid), dim=-1)   
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic
        
        for i in range(self.depth):
            x = self.layers1(x)
            if i != self.depth-1:
                x = F.gelu(x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)       
        return x
    
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



def rmse_loss(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    d2 = (x - y)**2 
    return torch.sqrt(torch.mean(d2))       


def loss_plot(Train_L2, Test_L2, mode, width, depth):
    x_axis = np.arange(0, len(Train_L2)) 
    fig = plt.figure(figsize=(20, 7.5)) 

    ax = plt.axes() 
    ax.plot(x_axis, Train_L2, label='Train L2', linewidth=2.0, color='blue')
    ax.plot(x_axis, Test_L2, label='Test L2', linewidth=2.0, color='green')
    ax.set_xlim(0,len(Train_L2)) 
    # ax.set_yscale('log')
    ax.xaxis.set_minor_locator(MultipleLocator(1)) 

    ax.set_xlabel('Epoch', fontdict={'family': 'Times New Roman', 'size': 20}) 
    ax.set_ylabel('Loss', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.xticks(fontname='Times New Roman', fontsize=16)
    plt.yticks(fontname='Times New Roman', fontsize=16)
    ax.legend(fancybox=True, framealpha=0.3, shadow=False, prop={'family': 'Times New Roman', 'size': 25})
    plt.title('mode:'+str(mode)+'\nwidth:'+str(width)+'\ndepth:'+str(depth))
    plt.show()
   
def save_data(Train_L2, Test_L2, mode1, width, depth):
    trial = 'm_'+str(mode1)+'_w_'+str(width)+'_d_'+str(depth)
    path = 'D:/LYK/gzb2yunchi/FNO/optuna/v/'+trial
    os.makedirs(path)
    np.savetxt(path+'/train.txt',Train_L2)
    np.savetxt(path+'/test.txt',Test_L2)
    return path


def objective(trial):
    mode1 = trial.suggest_int("mode", 3, 12, step=3)
    width = trial.suggest_int("width", 10, 40, step=10)
    depth = trial.suggest_int("depth", 1, 4, step=1)
    model = FNO2d(depth, mode1, mode1, width).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # myloss = LpLoss(size_average=False)
    pbar = range(epochs)
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)
    Train_L2 = []
    Test_L2 = []
    for ep in pbar:
        model.train()
        t1 = default_timer()
        train_l2 = 0
        idx_train = 0
        idx_test = 0
        for x, y, z in train_loader:
            # initial depth; BCs; velocity label
            x, y, z = x.cuda(), y.cuda(), z.cuda()
            optimizer.zero_grad()
            out = model(x, y)
            out = y_normalizer.decode(out)
            z = y_normalizer.decode(z)
            loss = rmse_loss(out.view(batch_size, -1), z.view(batch_size, -1))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()
            idx_train += 1
        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y, z in test_loader:
                x, y, z = x.cuda(), y.cuda(), z.cuda()
                out = model(x,y).view(batch_size2, S_x, S_y, 1)
                out = y_normalizer.decode(out)
                loss = rmse_loss(out.view(batch_size2, -1), z.view(batch_size2, -1))
                test_l2 += loss.item()
                idx_test += 1

    
        train_l2 /= ntrain
        test_l2 /= ntest
        Train_L2.append(train_l2)
        Test_L2.append(test_l2)
        t2 = default_timer()
        print('\nepoch: ', ep, '\ncompute time: ', t2-t1, '\ntrain L2: ', train_l2, '\ntest L2: ', test_l2, '\n', '='*30)
    loss_plot(Train_L2, Test_L2, mode1, width, depth)
    path = save_data(Train_L2, Test_L2, mode1, width, depth)
    torch.save({'model':model.state_dict()}, path +'/v.pth')
    return test_l2
        


################################################################
# configs
################################################################
DATA_PATH = 'D:/LYK/gzb2yunchi/data'
H_DATA_PATH = DATA_PATH + '/merge/elev1321.mat'
V_LABEL_PATH = DATA_PATH + '/merge/v1321.mat'
FLOW_PATH = DATA_PATH + '/discharge/discharge1321.txt'
UPWL_PATH = DATA_PATH + '/water level/upstream/yichang1321.txt'
DOWNWL_PATH = DATA_PATH + '/water level/downstream/yunchi1321.txt'


ntrain = 3000
ntest = 100
batch_size = 10
batch_size2 = 1
epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5
sub = 1
S_x = 760//sub
S_y = 20 // sub

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

runtime = np.zeros(2, )
t1 = default_timer()    


################################################################
# load data
################################################################
# Initial Condition
h_ini_reader = MatReader(H_DATA_PATH)
train_ini_h = h_ini_reader.read_field('h')[:ntrain,::sub,::sub]
test_ini_h = h_ini_reader.read_field('h')[ntrain:ntrain+ntest,::sub,::sub]

# Velocity Label
vtrain_reader = MatReader(V_LABEL_PATH)
train_label_v = vtrain_reader.read_field('v')[1:ntrain+1,::sub,::sub]
test_label_v = vtrain_reader.read_field('v')[ntrain+1:ntrain+ntest+1,::sub,::sub]

# Dimension expand
train_ini_h = train_ini_h.reshape(ntrain,S_x,S_y,1)
test_ini_h = test_ini_h.reshape(ntest,S_x,S_y,1)
train_label_v = train_label_v.reshape(ntrain,S_x,S_y,1)
test_label_v = test_label_v.reshape(ntest,S_x,S_y,1)

## Normalization
x_normalizer = UnitGaussianNormalizer(train_ini_h)
train_ini_h = x_normalizer.encode(train_ini_h)
test_ini_h = x_normalizer.encode(test_ini_h)

y_normalizer = UnitGaussianNormalizer(train_label_v)
train_label_v = y_normalizer.encode(train_label_v)

################################################################
## Boundary Conditions
################################################################
# upstream inflow condition
inflow = np.loadtxt(FLOW_PATH)
max_inflow, min_inflow = np.max(inflow), np.min(inflow)
inflow = (inflow - min_inflow) / (max_inflow - min_inflow)
inflow_a = inflow[1:ntrain+1]
inflow_u = inflow[ntrain+1:ntrain+ntest+1]
inflow_train = np.zeros((ntrain, S_x, S_y, 1))
inflow_test = np.zeros((ntest, S_x, S_y, 1))
for i in range(ntrain):
    inflow_train[i,:,:,0] = inflow_a[i]
for i in range(ntest):
    inflow_test[i,:,:,0] = inflow_u[i]
inflow_train = torch.tensor(inflow_train, dtype=torch.float)
inflow_test = torch.tensor(inflow_test, dtype=torch.float)

# downstream waterlevel condition
wl_train = np.zeros((ntrain, S_x, S_y, 1))
wl_test = np.zeros((ntest, S_x, S_y, 1))
downwl = np.loadtxt(DOWNWL_PATH)
max_downwl, min_downwl = np.max(downwl), np.min(downwl)
downwl = (downwl - min_downwl) / (max_downwl - min_downwl)
wl_a = downwl[1:ntrain+1]
wl_u = downwl[ntrain+1:ntrain+ntest+1]
for i in range(ntrain):
    wl_train[i,:,:,0] = wl_a[i]
for i in range(ntest):
    wl_test[i,:,:,0] = wl_u[i]
wl_train = torch.tensor(wl_train, dtype=torch.float)
wl_test = torch.tensor(wl_test, dtype=torch.float)

bc_train = torch.cat((inflow_train, wl_train), dim=-1) # concatenate training boundary condition
bc_test =  torch.cat((inflow_test, wl_test), dim=-1)   # concatenate testing boundary condition

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_ini_h, bc_train, train_label_v), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_ini_h, bc_test, test_label_v), batch_size=batch_size2, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1, 's')
y_normalizer.cuda()
 

# ################################################################
#  optuna
# ################################################################
search_space = {"mode": [3,6,9,12], "width": [10,20,30,40], "depth":[1,2,3,4]}
study = optuna.create_study(study_name='velocity',direction='minimize',storage='sqlite:///v.sqlite3', 
                            load_if_exists=True, sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, n_trials=64, n_jobs=1)

print('\nbest params = ',study.best_params)
print('\nbest trial value = ',study.best_trial.value)
df = study.trials_dataframe()


######################################## 
# 可视化命令
# Anaconda Prompt
# activate pytorch
# cd D:\LYK\gzb2yunchi\FNO\optuna
# optuna-dashboard sqlite:///v.sqlite3
######################################## 


# # parallel optuna
def optimize(n_trials):
    study = optuna.load_study(study_name='test', storage='sqlite:///db.sqlite3')
    study.optimize(objective, n_trials=n_trials)


study = optuna.create_study(study_name='test',direction='minimize',storage='sqlite:///db.sqlite3', load_if_exists=True)
r = Parallel(n_jobs=-1)([delayed(optimize)(10) for _ in range(10)])


