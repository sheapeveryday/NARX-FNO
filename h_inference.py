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
import scipy


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


def lossplot(Train_L2, Test_L2, learning_rate, scheduler_step, scheduler_gamma):
    x_axis = np.arange(0, len(Train_L2)) 
    fig = plt.figure(figsize=(20, 7.5)) 

    ax = plt.axes() 
    ax.plot(x_axis, Train_L2, label='Train L2', linewidth=2.0, color='blue')
    ax.plot(x_axis, Test_L2, label='Test L2', linewidth=2.0, color='green')
    ax.set_xlim(0,len(Train_L2)) 
    # ax.set_ylim(0,1) 
    ax.xaxis.set_minor_locator(MultipleLocator(1)) 

    ax.set_xlabel('Epoch', fontdict={'family': 'Times New Roman', 'size': 20}) 
    ax.set_ylabel('Loss', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.xticks(fontname='Times New Roman', fontsize=16) 
    plt.yticks(fontname='Times New Roman', fontsize=16)
    ax.legend(fancybox=True, framealpha=0.3, shadow=False, prop={'family': 'Times New Roman', 'size': 25}) 
    plt.title('learning rate:'+str(learning_rate)+'\nstep:'+str(scheduler_step)+'\ngamma:'+str(scheduler_gamma))
    plt.show()
   
    
DATA_PATH = os.getcwd()
H_DATA_PATH = DATA_PATH + '/merge/elev1321.mat'
H_LABEL_PATH = DATA_PATH + '/merge/h1321.mat'
data_reader = MatReader(H_DATA_PATH)
data_reader = data_reader.read_field('h')
train_ini_h = data_reader[:3000,:,:]
train_ini_h = train_ini_h.reshape(3000,760,20,1)
label_reader = MatReader(H_LABEL_PATH)
label_reader = label_reader.read_field('h')
train_label_h = label_reader[1:3001,:,:]
train_label_h = train_label_h.reshape(3000,760,20,1)
x_normalizer = UnitGaussianNormalizer(train_ini_h)
y_normalizer = UnitGaussianNormalizer(train_label_h)

################################################################
# configs
################################################################

TEST_H_DATA_PATH = DATA_PATH + '/depth/elev2021.mat'
TEST_H_LABEL_PATH = DATA_PATH + '/depth/h2021.mat'
FLOW_PATH = DATA_PATH + '/discharge/discharge2021.txt'
WATERLEVEL_PATH = DATA_PATH + '/water level/downstream/yunchi2021.txt'

BATHY_PATH = DATA_PATH + '/bathymetry/bathymetry.npy'
bathymetry = np.load(BATHY_PATH).reshape(1,760,20,1)
bathymetry = torch.tensor(bathymetry, dtype=torch.float)

batch_size = 1
sub = 1     
S_x = 760
S_y = 20    

depth = 4
modes = 9
width = 30


#%%
# ################################################################
# # load data
# ################################################################
initial_day = 221  # 8月10日
# initial_day = 319  # 11月16日

duration = 21 # lead time
end_day = initial_day + duration  

hdata_reader = MatReader(TEST_H_DATA_PATH)
data_ini_h = hdata_reader.read_field('h')[initial_day:end_day,::sub,::sub]
hlabel_reader = MatReader(TEST_H_LABEL_PATH)
label_h = hlabel_reader.read_field('h')[initial_day+1:end_day+1,::sub,::sub]

data_ini_h = data_ini_h.reshape(duration,S_x,S_y,1)
label_h = label_h.reshape(duration,S_x,S_y,1)
data_ini_h = x_normalizer.encode(data_ini_h)

inflow = np.loadtxt(FLOW_PATH)
inflow = (inflow - 5387) / (50820.83333-5387)
inflow_a = inflow[initial_day+1:end_day+1]
q_test = np.zeros((duration, S_x, S_y, 1))
for i in range(duration):
    q_test[i,:,:,0] = inflow_a[i]
q_test = torch.tensor(q_test, dtype=torch.float)

wl = np.loadtxt(WATERLEVEL_PATH)
wl = (wl - 38.29653707) / (50.36913386 - 38.29653707)
wl_a = wl[initial_day+1:end_day+1]
wl_test = np.zeros((duration, S_x, S_y, 1))
for i in range(duration):
    wl_test[i,:,:,0] = wl_a[i]
wl_test = torch.tensor(wl_test, dtype=torch.float)   

bc_test = torch.cat((q_test, wl_test), dim=-1)


test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_ini_h, bc_test, label_h), batch_size=batch_size, shuffle=False)


model = FNO2d(depth, modes, modes, width)
myloss = LpLoss(size_average=False)
state_dict = torch.load('D:/LYK/gzb2yunchi/FNO/mix fno/hICBC.pth')
model.load_state_dict(state_dict['model'])
pred = torch.zeros(label_h.shape)
result = np.ones((duration,S_x,S_y))
idx_pred = 0
Test_l2 = []
model.eval()
t1 = default_timer()
with torch.no_grad():
    for x, y, z in test_loader:     
        if idx_pred == 0:
            a = x
        else:
            a = pred[idx_pred-1].reshape(1,S_x,S_y,1)
            a += bathymetry
            tmp = a[0,:,:,0].numpy()
            a = x_normalizer.encode(a)
        out = model(a,y).view(batch_size, S_x, S_y, 1)
        out = y_normalizer.decode(out)        
        h_loss = rmse_loss(out.view(batch_size, -1), z.view(batch_size, -1)).item()
        pred[idx_pred,:,:] = out
        print(idx_pred, h_loss)
        result[idx_pred,:,:] = out[0,:,:,0].numpy()
        idx_pred += 1
t2 = default_timer()
print("time:", t2-t1) 
        
# Save file
path = 'D:/LYK/gzb2yunchi/FNO/inference/prediction/mat/h'
for i in range(duration):
    date = str(i+1)
    Ti = result[i,:,:]
    scipy.io.savemat(path+'/h'+date+'.mat', mdict={'pred': Ti})

   
# #%%
# # # # ################################################################
# # # # # assemble test
# # # # ################################################################


# initial_day = 212  
# # period = [3,7,15,21]  

# for d in range(21):
#     d += 1
#     inf_num = 365 - initial_day - d 
#     hresult = np.zeros((inf_num,d))   
#     fname = 'D:/LYK/gzb2yunchi/FNO/inference/uncertainty/h/err'+str(d)+'d.txt' 
#     for t in range(inf_num):
#         initial_day_iter = initial_day + t
#         end_day = initial_day_iter + d
#         test_ini = hdata_reader.read_field('h')[initial_day_iter:end_day,:,:]
#         test_h = hlabel_reader.read_field('h')[initial_day_iter+1:end_day+1,:,:]
        
#         inflow = np.loadtxt(FLOW_PATH)
#         inflow_a = inflow[initial_day_ite+1:end_day+1]
#         inflow_test = np.zeros((d, S_x, S_y, 1))
#         for i in range(d):
#             inflow_test[i,:,:,0] = inflow_a[i]
#         inflow_test = torch.tensor(inflow_test, dtype=torch.float)
        
#         wl = np.loadtxt(WATERLEVEL_PATH)
#         wl_a = wl[initial_day_iter+1:end_day+1]
#         wl_test = np.zeros((d, S_x, S_y, 1))
#         for i in range(d):
#             wl_test[i,:,:,0] = wl_a[i]
#         wl_test = torch.tensor(wl_test, dtype=torch.float)
        
#         test_ini = test_ini.reshape(d,S_x,S_y,1)
#         test_h = test_h.reshape(d,S_x,S_y,1)
#         test_ini = x_normalizer.encode(test_ini)
        
#         inflow_test = inflow_test.reshape(d,S_x,S_y,1)
#         wl_test = wl_test.reshape(d,S_x,S_y,1)
#         test_bc =  torch.cat((inflow_test, wl_test), dim=-1)
#         test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_h, test_bc, test_v), batch_size=1, shuffle=False)       
        
#         model = FNO2d(depth, modes, modes, width)
#         # myloss = LpLoss(size_average=False)
#         state_dict = torch.load('D:/LYK/gzb2yunchi/FNO/mix fno/hICBC.pth')
#         model.load_state_dict(state_dict['model'])    
#         pred = torch.zeros(test_v.shape)
#         idx_pred = 0
        
#         with torch.no_grad():
#             for x, y, z in test_loader:
#                 test_l2 = 0
#                 if idx_pred == 0:
#                     a = x
#                 else:
#                     a = pred[idx_pred-1].reshape(1,S_x,S_y,1)
#                     a += bathymetry
#                     tmp = a[0,:,:,0].numpy()
#                     a = x_normalizer.encode(a)
#                 out = model(a,y).view(batch_size, S_x, S_y, 1)
#                 out = y_normalizer.decode(out)
#                 pred[idx_pred,:,:,:] = out.reshape(1,S_x,S_y,1)
#                 h_loss = rmse_loss(out.view(batch_size, -1), z.view(batch_size, -1)).item()
#                 result[t, idx_pred] = h_loss
#                 idx_pred += 1
                
        
#     np.savetxt(fname, result)
           
    

