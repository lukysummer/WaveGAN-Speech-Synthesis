import numpy as np
import librosa
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


############################# 1. LOAD AUDIO DATA ##############################
from LoadData import LoadandProcessData

batch_size = 64
train_loader = LoadandProcessData(audio_dir = "drums/train/",
                                  batch_size = batch_size)
train_generator, n_batches = train_loader.batch_generator()

valid_loader = LoadandProcessData(audio_dir = "drums/valid/",
                                  batch_size = batch_size)
valid_generator, _ = valid_loader.batch_generator()



##################### 2. CONSTRUCT G & D MODEL INSTANCES ######################
from WaveGAN_models import waveganGenerator, waveganDiscriminator

G = waveganGenerator(d = 64).to(device)
D = waveganDiscriminator(d = 64).to(device)
G.load_state_dict(torch.load("G.pt"))
D.load_state_dict(torch.load("D.pt"))

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
save_every = 5
sample_every = 5

lr = 0.001
G_optimizer = optim.Adam(G.parameters(), lr, [0.5, 0.9])
D_optimizer = optim.Adam(D.parameters(), lr, [0.5, 0.9])

losses = []
G.train()
D.train()

for e in range(1, n_epochs+1):
    
    train_iter = iter(train_generator)
    valid_iter = iter(valid_generator)

    #for i in range(10): 
    for i in range(n_batches): 
        
        ##### 1. Obtain input x & latent vector z #####
        x = next(train_iter)['X']
        x = torch.from_numpy(x).float().to(device)
        
        valid_x = next(valid_iter)['X']   
        valid_x = torch.from_numpy(valid_x).float().to(device)
        
        ##### 2. Update D ######
        for _ in range(D_updates_per_G_update):
    
            z = np.random.uniform(-1, 1, size = (batch_size, 100))
            z = torch.from_numpy(z).float().to(device)
            
            D_optimizer.zero_grad()
            D_loss, D_wasserstein = DiscriminatorLoss(D, G, x, z, batch_size, device, 
                                                      compute_grad = True)
            D_optimizer.step()
            
            # D Validation #
            D_loss_valid, D_wasserstein_valid = DiscriminatorLoss(D, G, valid_x, z, batch_size, device, 
                                                                  compute_grad = False)
        
        ##### 3. Update G #####
        z = np.random.uniform(-1, 1, size = (batch_size, 100))
        z = torch.from_numpy(z).float().to(device)
   
        G_optimizer.zero_grad()
        G_loss = GeneratorLoss(G, D, z, device, compute_grad = True)
        G_optimizer.step()
        
        # G Validation #
        G_loss_valid = GeneratorLoss(G, D, z, device, compute_grad = False)

    ##### Monitor results at each epoch #####
    losses.append((D_loss.item(), G_loss.item()))
    print('Epoch [{:5d}/{:5d}] | D_loss: {:6.6f} | D_loss_valid: {:6.6f}'.format(
            e, n_epochs, D_loss.item(), D_loss_valid.item()))
    
    ##### Save the model regularly #####
    if (e % save_every) == 0:
        print("Saving models...")
        torch.save(G.state_dict(), "G" + "_after_"+str(n_epochs)+"_epochs.pt")
        torch.save(D.state_dict(), "D" + "_after_"+str(n_epochs)+"_epochs.pt")
    
    ##### Sample generated audio regularly #####
    if (e % sample_every) == 0:
        print("Saving sample audio...\n")
        G.eval()
        z = np.random.uniform(-1, 1, size = (1, 100))
        z = torch.from_numpy(z).float().to(device) 
        sound = G(z).detach().cpu().numpy()
        sound = sound.reshape(sound.shape[2],)
        librosa.output.write_wav("drums/sound"+"_after_"+str(n_epochs)+"_epochs.wav", sound, sr = 16000)