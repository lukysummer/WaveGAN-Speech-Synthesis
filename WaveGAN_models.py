import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

############################### GENERATOR MODEL ###############################
class waveganGenerator(nn.Module):
    
    def __init__(self, d):  # d = model_size
        
        super(waveganGenerator, self).__init__()
        self.d = d
        self.dense = nn.Linear(100, 256*d)
        self.deconv1 = nn.ConvTranspose1d(16*d, 8*d, kernel_size=24, stride=4, padding=11)
        self.deconv2 = nn.ConvTranspose1d(8*d, 4*d, kernel_size=24, stride=4, padding=11)
        self.deconv3 = nn.ConvTranspose1d(4*d, 2*d, kernel_size=24, stride=4, padding=11)
        self.deconv4 = nn.ConvTranspose1d(2*d, d, kernel_size=24, stride=4, padding=11)
        self.deconv5 = nn.ConvTranspose1d(d, 1, kernel_size=24, stride=4, padding=11)
        
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose1d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
        
    
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



############################# DISCRIMINATOR MODEL #############################      
class waveganDiscriminator(nn.Module):
    
    def __init__(self, d, shift_factor = 2, alpha = 0.2):
        
        super(waveganDiscriminator, self).__init__()
        self.d = d
        self.alpha = alpha
        self.shift_factor = shift_factor
        
        self.conv1 = nn.Conv1d(1, d, kernel_size = 25, stride = 4, padding = 11)
        self.conv2 = nn.Conv1d(d, 2*d, kernel_size = 25, stride = 4, padding = 11)
        self.conv3 = nn.Conv1d(2*d, 4*d, kernel_size = 25, stride = 4, padding = 11)
        self.conv4 = nn.Conv1d(4*d, 8*d, kernel_size = 25, stride = 4, padding = 11)
        self.conv5 = nn.Conv1d(8*d, 16*d, kernel_size = 25, stride = 4, padding = 11)
        self.dense = nn.Linear(256*d, 1)
        
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
        
        
    ###### PHASE SHUFFLE ######
    def phase_shuffle(self, audio_batch):
        '''
        Performs phase shuffling (to be used by Discriminator ONLY) by shifting 
        each audio signal by a random number of sample in range 
        [-shift_factor, shift_factor]
        '''
        audio_batch = audio_batch.detach().numpy().astype(int)
        
        for i, audio in enumerate(audio_batch):
            print(audio[0].shape)
            shift = np.random.choice(range(-1*self.shift_factor, self.shift_factor+1))
            if shift > 0: # Remove the last k values & insert k left-paddings 
                audio_batch[i] = np.pad(audio[0][:-shift], (shift, 0), mode = 'reflect').reshape(1, -1)
            else:         # Remove the first k values & insert k right-paddings 
                audio_batch[i] = np.pad(audio[0][abs(shift):], (0, abs(shift)), mode = 'reflect').reshape(1, -1)
        
        audio_batch = torch.from_numpy(audio_batch).float().to(device)
        
        return audio_batch
        
        
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