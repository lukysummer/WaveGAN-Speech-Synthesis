import torch
from torch import nn
import torch.nn.functional as F

############################### GENERATOR MODEL ###############################
class waveganGenerator(nn.Module):
    
    def __init__(self, d):  # d = model_size
        
        super(waveganGenerator, self).__init__()
        self.d = d
        self.dense = nn.Linear(100, 256*d)
        self.deconv1 = nn.ConvTranspose1d(16*d, 8*d, kernel_size = 24, stride = 4, padding = 10)
        self.deconv2 = nn.ConvTranspose1d(8*d, 4*d, kernel_size = 24, stride = 4, padding = 10)
        self.deconv3 = nn.ConvTranspose1d(4*d, 2*d, kernel_size = 24, stride = 4, padding = 10)
        self.deconv4 = nn.ConvTranspose1d(2*d, d, kernel_size = 24, stride = 4, padding = 10)
        self.deconv5 = nn.ConvTranspose1d(d, 1, kernel_size = 24, stride = 4, padding = 10)
        
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose1d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module)
        
        
    def forward(self, z):            # z shape: (64, 100)

        z = self.dense(z)                      #(64, 256d)
        z = z.reshape((-1, 16*self.d, 16))     #(64, 16d, 16)
        z = F.relu(z)

        z = F.relu(self.deconv1(z))            #(64, 8d, 64)
        z = F.relu(self.deconv2(z))            #(64, 4d, 256)
        z = F.relu(self.deconv3(z))            #(64, 2d, 1024)
        z = F.relu(self.deconv4(z))            #(64, d, 4096)
        output = torch.tanh(self.deconv5(z))   #(64, 1, 16384)
        
        return output



################### PHASE SHUFFLE MODEL FOR DISCRIMINATOR #####################
############# (taken from:  https://github.com/jtcramer/wavegan) ##############
class PhaseShuffle(nn.Module):
    '''
    Performs phase shuffling (to be used by Discriminator ONLY) by: 
       -Shifting feature axis of a 3D tensor by a random integer in [-n, n] 
       -Performing reflection padding where necessary
    
    If batch shuffle = True, only a single shuffle is applied to the entire 
    batch, rather than each sample in the batch.
    '''
    def __init__(self, shift_factor):
        
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor
        
        
    def forward(self, x):    # x shape: (64, 1, 16384)
        # Return x if phase shift is disabled
        if self.shift_factor == 0:
            return x
        
        # k_list = [-shift_factor, -shift_factor + 1, ..., 0, ..., shift_factor - 1, shift_factor]
        k_list = torch.Tensor(x.shape[0]).random_(0, 2*self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)
        
        k_map = {}  # 5 items
        for sample_idx, k in enumerate(k_list):
            k = int(k) 
            if k not in k_map:
                k_map[k] = []
            
            k_map[k].append(sample_idx)
            
        shuffled_x = x.clone()
        
        for k, sample_idxs in k_map.items():
            if k > 0:   # 1. Remove the last k values & 2. Insert k left-paddings 
                shuffled_x[sample_idxs] = F.pad(x[sample_idxs][..., :-k], 
                                                pad = (k, 0), 
                                                mode = 'reflect')
            
            else:       # 1. Remove the first k values & 2. Insert k right-paddings 
                shuffled_x[sample_idxs] = F.pad(x[sample_idxs][..., abs(k):], 
                                                  pad = (0, abs(k)), 
                                                  mode = 'reflect')
                    
        assert shuffled_x.shape == x.shape, "{}, {}".format(shuffled_x.shape, x.shape)
        
        return shuffled_x
        
     

############################# DISCRIMINATOR MODEL #############################      
class waveganDiscriminator(nn.Module):
    
    def __init__(self, d, shift_factor = 2, alpha = 0.2):
        
        super(waveganDiscriminator, self).__init__()
        self.d = d
        self.alpha = alpha
        
        self.conv1 = nn.Conv1d(1, d, kernel_size = 24, stride = 4, padding = 10)
        self.conv2 = nn.Conv1d(d, 2*d, kernel_size = 24, stride = 4, padding = 10)
        self.conv3 = nn.Conv1d(2*d, 4*d, kernel_size = 24, stride = 4, padding = 10)
        self.conv4 = nn.Conv1d(4*d, 8*d, kernel_size = 24, stride = 4, padding = 10)
        self.conv5 = nn.Conv1d(8*d, 16*d, kernel_size = 24, stride = 4, padding = 10)
        self.dense = nn.Linear(256*d, 1)
        
        self.phase_shuffle = PhaseShuffle(shift_factor)
        
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose1d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module)
        
        
    def forward(self, x):                                # x shape: (64, 1, 16384)

        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha) #(64, d, 4096)
        x = self.phase_shuffle(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha) #(64, 2d, 1024)
        x = self.phase_shuffle(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha) #(64, 4d, 256)
        x = self.phase_shuffle(x)
        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha) #(64, 8d, 64)
        x = self.phase_shuffle(x)
        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha) #(64, 16d, 16)
        
        x = x.reshape((-1, 256*self.d))                            #(64, 256d)
        output = self.dense(x)                                     #(64, 1)
        
        return output
