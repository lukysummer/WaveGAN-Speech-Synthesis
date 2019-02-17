import numpy as np
import os

import librosa
from scipy.io import wavfile
import pescador


def decode_audio(filepath, _sample_rate = 16000):
    # 1. Read in the wavfile
    #sample_rate, signal = wavfile.read(filepath)
    signal, sample_rate = librosa.core.load(filepath, 
                                            sr = _sample_rate, 
                                            mono = False)
    
    # 2. Change the datatype as float
    if signal.dtype == np.int16:
        signal = signal.astype(np.float32)
        signal /= 32768.
        
    # 3. Reshape the signal
    if signal.ndim == 1:
        n_channels = 1
    else:
        n_channels = signal.shape[1]
        
    signal = signal.reshape(n_channels, signal.shape[0])
        
    # 4. Normalize the signal
    signal /= np.max(np.abs(signal))
    
    yield {'X': signal}
    
    #return signal


def get_all_audio_filepaths(audio_dir):
    
    return [os.path.join(root, fname)
            for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names
            if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'))]



def batch_generator(audio_filepath_list, batch_size):
    
    streamers = []
    for audio_filepath in audio_filepath_list:
        s = pescador.Streamer(decode_audio, audio_filepath)
        streamers.append(s)
    
    print("Number of Training Examples: ", len(streamers))
    n_batches = len(streamers)//batch_size
    # takes N streamers, and samples from them equally, guaranteeing all N streamers to be “active”.
    mux = pescador.ShuffledMux(streamers)
    batch_gen = pescador.buffer_stream(mux, batch_size)

    return batch_gen, n_batches



import torch
from torch import nn
import torch.nn.functional as F

class waveganGenerator(nn.Module):
    
    def __init__(self, d):  # d = model_size
        
        super(waveganGenerator, self).__init__()
        
        self.d = d
        self.dense = nn.Linear(100, 256*d)
        self.deconv1 = nn.ConvTranspose1d(16*d, 8*d, kernel_size = 25, stride = 4)
        self.deconv2 = nn.ConvTranspose1d(8*d, 4*d, kernel_size = 25, stride = 4)
        self.deconv3 = nn.ConvTranspose1d(4*d, 2*d, kernel_size = 25, stride = 4)
        self.deconv4 = nn.ConvTranspose1d(2*d, d, kernel_size = 25, stride = 4)
        self.deconv5 = nn.ConvTranspose1d(d, 1, kernel_size = 25, stride = 4)
        
    def forward(self, z):
        
        dense_out = self.dense(z).reshape((-1, 16, 16*self.d))
        dense_out = F.relu(dense_out)
        
        deconv_out = F.relu(self.deconv1(dense_out))
        deconv_out = F.relu(self.deconv2(deconv_out))
        deconv_out = F.relu(self.deconv3(deconv_out))
        deconv_out = F.relu(self.deconv4(deconv_out))
        deconv_out = torch.tanh(self.deconv5(deconv_out))
        
        return deconv_out
        
     
        
class waveganDiscriminator(nn.Module):
    
    def __init__(self, d):
        
        super(waveganDiscriminator, self).__init__()
        
        self.d = d
        self.conv1 = nn.Conv1d(1, d, kernel_size = 25, stride = 4)
        self.conv2 = nn.Conv1d(d, 2*d, kernel_size = 25, stride = 4)
        self.conv3 = nn.Conv1d(2*d, 4*d, kernel_size = 25, stride = 4)
        self.conv4 = nn.Conv1d(4*d, 8*d, kernel_size = 25, stride = 4)
        self.conv5 = nn.Conv1d(8*d, 16*d, kernel_size = 25, stride = 4)
        self.dense = nn.Linear(256*d, 1)
        
    def forward(self, x):
        #print("x: ", x.shape)
        conv_out = F.leaky_relu(self.conv1(x), negative_slope = 0.2)
        conv_out = F.leaky_relu(self.conv2(conv_out), negative_slope = 0.2)
        conv_out = F.leaky_relu(self.conv3(conv_out), negative_slope = 0.2)
        conv_out = F.leaky_relu(self.conv4(conv_out), negative_slope = 0.2)
        conv_out = F.leaky_relu(self.conv5(conv_out), negative_slope = 0.2)
        
        conv_out = conv_out.reshape((-1, 256*self.d))
        dense_out = self.dense(conv_out)
        
        return dense_out
        

G = waveganGenerator(d = 1)
D = waveganDiscriminator(d = 1)
print("G & D Models Constructed.\n")

from torch import optim

n_epochs = 1
lr = 0.001
update_D_every = 1
G_optimizer = optim.Adam(G.parameters(), lr, [0.5, 0.9])
D_optimizer = optim.Adam(D.parameters(), lr, [0.5, 0.9])


batch_size = 64
train_paths = get_all_audio_filepaths("drums/train/")
train_generator, n_batches = batch_generator(train_paths, batch_size) 

losses = []
G.train()
D.train()

for e in range(1, n_epochs+1):
    
    train_iter = iter(train_generator)
    #batch = next(train_iter)['X']

    for i in range(n_batches): # range(n_batches)
        
        x = next(train_iter)['X']
        x = torch.from_numpy(x).float()
        
        z = np.random.uniform(-1, 1, size = (batch_size, 100))
        z = torch.from_numpy(z).float()

        ### Update G ###
        G.zero_grad()
        G_z = G(z)
        D_Gz = D(G_z)
        
        G_optimizer.zero_grad()
        G_loss = torch.mean((D_Gz)**2)
        G_loss.backward()
        G_optimizer.step()
        
        if i % update_D_every == 0:
            ### Update D ###
            z = np.random.uniform(-1, 1, size = (batch_size, 100))
            z = torch.from_numpy(z).float()
            
            D.zero_grad()
    
            G_z = G(z)
            D_Gz = D(G_z)
            D_x = D(x)
            
            D_real_loss =  torch.mean(D_x**2)
            D_fake_loss = torch.mean((D_Gz - 0.9)**2)
            D_loss = D_real_loss + D_fake_loss
        
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()
        
        
    losses.append((D_loss.item(), G_loss.item()))
    print('Epoch [{:5d}/{:5d}] | D_loss: {:6.6f} | G_loss: {:6.6f}'.format(
            e, n_epochs, D_loss.item(), G_loss.item()))


G.eval()
z = np.random.uniform(-1, 1, size = (1, 100))
z = torch.from_numpy(z).float()   
sound = G(z).detach().numpy()
sound = sound.reshape(sound.shape[2],)
librosa.output.write_wav("drums/sound" + n_epochs + ".wav", sound, sr = 16000)
     
        
        
                     