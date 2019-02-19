import numpy as np
import librosa
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

from WaveGAN_models import waveganGenerator

n_epochs = 10

G = waveganGenerator(d = 64).to(device)
G.load_state_dict(torch.load("G" + "_after_" + str(n_epochs) + "_epochs.pt"))


print("Saving sample audio...\n")
G.eval()
z = np.random.uniform(-1, 1, size = (1, 100))
z = torch.from_numpy(z).float().to(device) 
sound = G(z).detach().cpu().numpy()
sound = sound.reshape(sound.shape[2],)
librosa.output.write_wav("drums/sound_generated_after_"+str(n_epochs)+"_epochs.wav", sound, sr = 16000)