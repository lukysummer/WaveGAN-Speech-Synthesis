import numpy as np
import os

import librosa
import pescador

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


########################### 1. READ IN AUDIO DATA #############################
def decode_audio(filepath, _sample_rate = 16000, audio_length = 16384):
    '''
    Reads in an audio file and pre-processes it
    : param filepath: path to the audio file
    : param _sampel_rate: sample rate of the audio file [samples/second]
    '''
    ##### 1. Read in the wavfile using librosa library #####
    signal, sample_rate = librosa.core.load(filepath, 
                                            sr = _sample_rate, 
                                            mono = False)
    
    ##### 2. Change the datatype as float #####
    if signal.dtype == np.int16:
        signal = signal.astype(np.float32)
        signal /= 32768.
        
    ##### 3. Adjust the length of the audio file to audio_length #####
    if len(signal) < audio_length:
        total_padding = audio_length - len(signal)
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        
        signal = np.pad(signal, (left_padding, right_padding), mode = 'constant')
    else:
        signal = signal[:audio_length]
             
    ##### 4. Reshape the signal #####
    n_channels = 1 if signal.ndim == 1 else signal.shape[1]        
    signal = signal.reshape(n_channels, signal.shape[0])
        
    ##### 5. Normalize the signal #####
    signal /= np.max(np.abs(signal))
    
    yield {'X': signal}


def get_all_audio_filepaths(audio_dir):
    '''
    Returns a list of audio file paths 
    '''
    return [os.path.join(root, fname)
            for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names
            if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'))]


def batch_generator(audio_filepath_list, batch_size):
    '''
    Generates batches of audios
    : param audio_filepath_list: list of audio file paths
    : param batch_size: desired batch size 
    '''
    streamers = []
    for audio_filepath in audio_filepath_list:
        s = pescador.Streamer(decode_audio, audio_filepath)
        streamers.append(s)
    
    print("Number of Training Examples: \n", len(streamers))
    n_batches = len(streamers)//batch_size
    mux = pescador.ShuffledMux(streamers)
    batch_gen = pescador.buffer_stream(mux, batch_size)

    return batch_gen, n_batches


batch_size = 64
train_paths = get_all_audio_filepaths("sc09/train/")
train_generator, n_batches = batch_generator(train_paths, batch_size) 



########################## 2. BUILD GENERATOR MODEL ###########################
from torch import nn
import torch.nn.functional as F

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



############### 2. BUILD PHASE SHUFFLE MODEL FOR DISCRIMINATOR ################
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
        
     

######################### 4. BUILD DISCRIMINATOR MODEL ########################      
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
        


##################### 5. CONSTRUCT G & D MODEL INSTANCES ######################
G = waveganGenerator(d = 64).to(device)
D = waveganDiscriminator(d = 64).to(device)
print("G & D Models Constructed.\n")



########################## 6. DEFINE GRADIENT PENALTY #########################
from torch.autograd import Variable, grad

# Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
def gradientPenalty(D, x, z, batch_size, device, lmbda = 10.0):
    ''' lmbda : gradient penalty regularization factor '''
    ### 1. Compute interpolation factors
    alpha = torch.rand(batch_size, 1, 1) # x shape: (batch_size, 1, signal_length)
    alpha = alpha.expand(x.size()) # duplicate the same random # signal_length times
    alpha = alpha.to(device)
    
    ### 2. Interpolate between real & fake data
    interpolates = alpha * x + ((1 - alpha) * z)
    interpolates = interpolates.to(device)
    interpolates = Variable(interpolates, requires_grad = True)
    
    ### 3. Evaluate D
    D_interpolates = D(interpolates)
    
    ### 4. Obtain gradients of D w.r.t. inputs x
    gradients = grad(inputs = interpolates,
                     outputs = D_interpolates,
                     grad_outputs = torch.ones(D_interpolates.size()).to(device),
                     create_graph = True,
                     retain_graph = True,
                     only_inputs = True)[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmbda
    
    return gradient_penalty
    


######################### 7. DEFINE DISCRIMINATOR LOSS ########################
def DiscriminatorLoss(D, G, x, z, batch_size, device, compute_grad = False):
    
    ##### 1. Loss from Real Audio, x #####
    D.zero_grad()
    D_real = D(x)
    real_loss = D_real.mean()
    
    if compute_grad:
        real_loss.backward(torch.tensor(-1.0).to(device))  # -1 since want to maximize the real_loss
      
    ##### 2. Loss from Fake (generated) Audio, G(z) #####
    G_out = Variable(G(z).data)
    D_fake = D(G_out)
    fake_loss = D_fake.mean()
    
    if compute_grad:
        fake_loss.backward()     # want to minimize the fake loss
        
    ##### 3. Gradient Penalty #####
    gradient_penalty = gradientPenalty(D, x.data, G_out.data, batch_size, device)

    if compute_grad:
        gradient_penalty.backward()
    
    ##### 4. Calculate final 2 losses #####
    D_loss = fake_loss - real_loss + gradient_penalty
    D_Wasserstein = real_loss - fake_loss
    
    return D_loss, D_Wasserstein



########################### 8. DEFINE GENERATOR LOSS ##########################
def GeneratorLoss(G, D, z, device, compute_grad = False):
    G.zero_grad()
    G_out = G(z)
    D_fake = D(G_out)
    fake_loss = D_fake.mean()
    
    if compute_grad:
        fake_loss.backward(torch.tensor(-1.0).to(device))    # maximize
        
    G_loss = -1 * fake_loss   # * -1 because want to maximize the loss 
    
    return G_loss
    
    

############################# 9. TRAIN THE NETWORK ############################
from torch import optim

n_epochs = 10
D_updates_per_G_update = 5  # For each iteration, update G 1 time & update D 5 times
save_every = 10
sample_every = 10

lr = 0.001
G_optimizer = optim.Adam(G.parameters(), lr, [0.5, 0.9])
D_optimizer = optim.Adam(D.parameters(), lr, [0.5, 0.9])

losses = []
G.train()
D.train()

for e in range(1, n_epochs+1):
    
    train_iter = iter(train_generator)

    #for i in range(10): 
    for i in range(n_batches): 
        
        ##### 1. Obtain input x & latent vector z #####
        x = next(train_iter)['X']
        x = torch.from_numpy(x).float().to(device)
        
        z = np.random.uniform(-1, 1, size = (batch_size, 100))
        z = torch.from_numpy(z).float().to(device)
   
        ##### 2. Update D ######
        for _ in range(D_updates_per_G_update):
    
            z = np.random.uniform(-1, 1, size = (batch_size, 100))
            z = torch.from_numpy(z).float()
            
            D_optimizer.zero_grad()
            D_loss, D_wasserstein = DiscriminatorLoss(D, G, x, z, batch_size, device, 
                                                      compute_grad = True)
            D_optimizer.step()
            
            # D Validation #
            D_loss_valid, D_wasserstein_valid = DiscriminatorLoss(D, G, x, z, batch_size, device, 
                                                                  compute_grad = False)
            '''
            D.zero_grad()
            G_z = G(z)
            D_Gz = D(G_z)
            D_x = D(x)
            D_real_loss =  torch.mean(D_x**2)
            D_fake_loss = torch.mean((D_Gz - 0.9)**2)
            D_loss = D_real_loss + D_fake_loss 
            D_loss.backward()
        '''
        
        ##### 3. Update G #####
        G_optimizer.zero_grad()
        G_loss = GeneratorLoss(G, D, z, device, compute_grad = True)
        G_optimizer.step()
        
        # G Validation #
        G_loss_valid = GeneratorLoss(G, D, z, device, compute_grad = False)
        '''
        G.zero_grad()
        G_z = G(z)
        D_Gz = D(G_z)
        G_loss = torch.mean((D_Gz)**2)
        G_loss.backward()
        '''
    ##### Monitor results at each epoch #####
    losses.append((D_loss.item(), G_loss.item()))
    print('Epoch [{:5d}/{:5d}] | D_loss: {:6.6f} | D_loss_valid: {:6.6f}'.format(
            e, n_epochs, D_loss.item(), D_loss_valid.item()))
    
    ##### Save the model regularly #####
    if (e % save_every) == 0:
        torch.save(G.state_dict(), "G.pt")
        torch.save(D.state_dict(), "D.pt")
    
    ##### Sample generated audio regularly #####
    if (e % sample_every) == 0:
        G.eval()
        z = np.random.uniform(-1, 1, size = (1, 100))
        z = torch.from_numpy(z).float()   
        sound = G(z).detach().numpy()
        sound = sound.reshape(sound.shape[2],)
        librosa.output.write_wav("drums/sound" + str(n_epochs) + ".wav", sound, sr = 16000)
    

     
'''
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses (Epochs 1-50)")
plt.legend()  
'''      
