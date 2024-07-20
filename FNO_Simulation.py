# import all modules and our model layers
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys,os 
import matplotlib.pyplot as plt
from utilities3 import *
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from tensorflow.keras.optimizers import Adam
torch.manual_seed(3407)
np.random.seed(0)
torch.set_num_threads(1)

################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(8, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        
        
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
       
        
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
       
        
        
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 25, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic
        
        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        
        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)     
        
        
        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)




TRAIN_PATH_Niigata = '20240626Niigata实验数据/20240626SlicedNiigata_42times_40timeOverlap_RandomTrain_64x64.npy'
TRAIN_SDF_Niigata = '20240626Niigata实验数据/20240626SlicedNiigata_42times_40timeOverlap_RandomTrain_64x64_SDF.npy'

TEST_PATH_Niigata = '202407写论文需要的全部工作/实验结果/20240701SlicedNiigata_1200times_00timeOverlap_WestTestUpDown_64x64_16个拼出原图.npy'
TEST_SDF_Niigata = '202407写论文需要的全部工作/实验结果/20240701SlicedNiigata_1200times_00timeOverlap_WestTestUpDown_64x64_16个拼出原图_SDF.npy'



batch_size = 500
epochs = 100
learning_rate = 0.001

scheduler_step = 50
scheduler_gamma = 0.5
modes = 26
width = 40

sub = 1
S = 64 // sub
T_in = 5
T_out = 25


model_path = '20240626Niigata实验结果/20240626Gelu_NiigataWestTestOnUDRotate_modes'+str(modes)+'_width'+str(width)+'_epoch'+str(epochs)+'_input'+str(T_in)+'_output'+str(T_out)+'_40timeOverlap_'+str(S)+'x'+str(S)+'_WITHSDF'


train_err_path = model_path+'_trainErr.txt'
test_err_path = model_path+'_testErr.txt'
image_path = '20240626MontrealIMG/'+model_path




data_train_nii = np.load(TRAIN_PATH_Niigata)
data_test_nii = np.load(TEST_PATH_Niigata)
data_train = data_train_nii
data_test = data_test_nii
ntrain = data_train.shape[0]
ntest = data_test.shape[0]
print(data_test.shape)


train_a = data_train[:, :, :, :T_in]
train_u = data_train[:, :, :, T_in:T_in+T_out]
test_a = data_test[:, :, :, :T_in]
test_u = data_test[:, :, :, T_in:T_in+T_out]


# Torchlize
train_a = torch.Tensor(train_a)
train_u = torch.Tensor(train_u)
test_a = torch.Tensor(test_a)
test_u = torch.Tensor(test_u)

#normalize data
a_normalizer = GaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)
y_normalizer = GaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)



sdfTrainNii = np.load(TRAIN_SDF_Niigata)
sdfTestNii = np.load(TEST_SDF_Niigata)
sdfTrain = sdfTrainNii
sdfTest = sdfTestNii
sdf_train = torch.from_numpy(sdfTrain)
sdf_test = torch.from_numpy(sdfTest)

#normalize sdf
sdf_a_normalizer = GaussianNormalizer(sdf_train)
sdf_train = sdf_a_normalizer.encode(sdf_train)
sdf_test = sdf_a_normalizer.encode(sdf_test)

#squeeze to make sdf dimension suiteable for us
sdf_train = sdf_train.unsqueeze(-1).float()
sdf_test = sdf_test.unsqueeze(-1).float()

train_a = torch.cat((train_a, sdf_train), -1)
test_a = torch.cat((test_a, sdf_test), -1)

# ok we make our loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=16, shuffle=False, pin_memory=True)
batch_total = round(test_u.shape[0]/batch_size)
del  data_train_nii, train_a, train_u, test_a, test_u, sdf_train, sdf_test


#training error and testing error and epochs
train_mse_err = torch.tensor([])
test_l2_err = torch.tensor([])
train_l2_err = torch.tensor([])
train_rela_err = torch.tensor([])
test_rela_err = torch.tensor([])

epochs_done = 0

model = FNO2d(modes, modes, width).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma, verbose=True)



checkpoint = torch.load(model_path+'.pt', map_location=torch.device('cuda:1'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epochs_done = checkpoint['epoch']

# train_mse_err = torch.load('20240128FNO_test30_modes26width40_RelaLoss_00timeOverlap_Random_MTL500x500x2000_T_in1_T_out29_ep_50_Train_mse_arr')
# train_l2_err = torch.load('20240128FNO_test30_modes26width40_RelaLoss_00timeOverlap_Random_MTL500x500x2000_T_in1_T_out29_ep_50_Train_l2_arr')
# test_l2_err = torch.load('20240128FNO_test30_modes26width40_RelaLoss_00timeOverlap_Random_MTL500x500x2000_T_in1_T_out29_ep_50_Test_l2_arr')
# train_rela_err = torch.load('20240128FNO_test30_modes26width40_RelaLoss_00timeOverlap_Random_MTL500x500x2000_T_in1_T_out29_ep_50_Train_rela')
# test_rela_err = torch.load('20240128FNO_test30_modes26width40_RelaLoss_00timeOverlap_Random_MTL500x500x2000_T_in1_T_out29_ep_50_Test_rela')



import sys
from IPython.display import clear_output
device = 'cuda:1'
from tqdm import tqdm
myloss = LpLoss(size_average=False)
y_normalizer.to(device)
model = model.to(device)
print(batch_total)
showloss = []
# start
for ep in tqdm(range(epochs_done, epochs)):
    model.train() 
    t1 = default_timer() 
    train_mse = 0
    train_l2 = 0
    train_rela = 0  
    test_rela = 0
    
    total_rela = []
    
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device) 
        
        optimizer.zero_grad() 

        out = model(x) 

        mse = F.mse_loss(out, y, reduction='mean')

        y = y_normalizer.decode(y)
        out = y_normalizer.decode(out)
        l2 = myloss(out.contiguous().view(out.shape[0], -1), y.contiguous().view(y.shape[0],-1))
        train_mse += mse.item()
        train_l2 += l2.item()
        relative_error_train = torch.mean(torch.sqrt(torch.mean(torch.square(out - y), axis = (1,2)) / torch.mean(torch.square(y), axis = (1,2))))
        relative_error_train.backward()
        print("train relative error")
        print(relative_error_train)
        train_rela += relative_error_train.item()
        
        optimizer.step()
    scheduler.step()
    
    model.eval()
    test_l2 = 0.0
    
    totaltttList = []
    count = 0
    with torch.no_grad():
        for x, y in test_loader:
            tttList = []
            x, y = x.to(device), y.to(device)
            y = y[:,:,:,:T_out]
            out = model(x).view(x.shape[0], S, S, T_out)
            count = count+1
            out = y_normalizer.decode(out)
            for homme in range(T_out):
                train_step_rela = torch.mean(torch.sqrt(torch.mean(torch.square(out[:,:,:,homme] - y[:,:,:,homme]), axis = (1,2)) / torch.mean(torch.square(y[:,:,:,homme]), axis = (1,2))))
                tttList.append(train_step_rela)
            totaltttList.append(tttList)
            
            relative_error_test = torch.mean(torch.sqrt(torch.mean(torch.square(out - y), axis = (1,2)) / torch.mean(torch.square(y), axis = (1,2))))
            print("test relative error")
            print(relative_error_test)
            test_rela += relative_error_test.item()
    
   
    hoshiList = []
    for index in range(T_out): 
        tmpval = 0
        for yndex in range(batch_total): 
            tmpval += totaltttList[yndex][index]
        tmpval /= batch_total
        tmpval = tmpval.cpu().numpy()
        hoshiList.append(tmpval)
    print(hoshiList)
 
    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest
    train_rela /= len(train_loader)
    test_rela /= len(test_loader)

    train_mse_err = torch.cat((train_mse_err, torch.tensor([train_mse])), -1)
    train_l2_err = torch.cat((train_l2_err, torch.tensor([train_l2])), -1)
    test_l2_err = torch.cat((test_l2_err, torch.tensor([test_l2])), -1)
    train_rela_err = torch.cat((train_rela_err, torch.tensor([train_rela])),-1)
    test_rela_err = torch.cat((test_rela_err, torch.tensor([test_rela])),-1)
    
    
    t2 = default_timer()
    print(f'{ep}, {t2-t1:.2f}, {train_mse:.5f}, {train_l2:.5f}, {test_l2:.5f}, {train_rela:.5f}, {test_rela:.5f}')
    if ep%5 == 0:
      torch.save({
            'epoch': ep+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }, model_path+'.pt')
    torch.save(train_mse_err, '20240626Niigata实验结果/Gelu20240626NiigataWestDataTestOnUDRotate64x64_train70_test30_modes'+str(modes)+'width'+str(width)+'_RelaLoss_40timeOverlap_Random_T_in'+str(T_in)+'_T_out'+str(T_out)+'_ep_'+str(epochs)+'_Train_mse_arr')#manually named
    torch.save(train_l2_err, '20240626Niigata实验结果/Gelu20240626NiigataWestDataTestOnUDRotate64x64_train70_test30_modes'+str(modes)+'width'+str(width)+'_RelaLoss_40timeOverlap_Random_T_in'+str(T_in)+'_T_out'+str(T_out)+'_ep_'+str(epochs)+'_Train_l2_arr')
    torch.save(test_l2_err, '20240626Niigata实验结果/Gelu20240626NiigataWestDataTestOnUDRotate64x64_train70_test30_modes'+str(modes)+'width'+str(width)+'_RelaLoss_40timeOverlap_Random_T_in'+str(T_in)+'_T_out'+str(T_out)+'_ep_'+str(epochs)+'_Test_l2_arr')
    torch.save(train_rela_err, '20240626Niigata实验结果/Gelu20240626NiigataWestDataTestOnUDRotate64x64_train70_test30_modes'+str(modes)+'width'+str(width)+'_RelaLoss_40timeOverlap_Random_T_in'+str(T_in)+'_T_out'+str(T_out)+'_ep_'+str(epochs)+'_Train_rela')
    torch.save(test_rela_err, '20240626Niigata实验结果/Gelu2024062NiigataWestDataTestOnUDRotate64x64_train70_test30_modes'+str(modes)+'width'+str(width)+'_RelaLoss_40timeOverlap_Random_T_in'+str(T_in)+'_T_out'+str(T_out)+'_ep_'+str(epochs)+'_Test_rela')
       
    print(f'saved epoch {ep} successfully!')
    
    
