import numpy as np
import os
import librosa
import torch
from torch.utils.data import TensorDataset, DataLoader

############################ READ IN AUDIO DATA ###############################
class LoadandProcessData():
    
    def __init__(self, audio_dir, batch_size, sample_rate = 16000, audio_length = 16384):
        '''
        : param audio_dir: directory containing the entire list of audio file paths
        : param batch_size: desired batch size 
        : param sample_rate: sample rate of the audio file [samples/sec]
        : param audio_length: length of audio in [samples/sec] to be inputted into the GAN model
        '''
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        

    def decode_audio(self, filepath):
        '''
        Reads in an audio file and pre-processes it
        : param filepath: path to the audio file
        '''
        ##### 1. Read in the wavfile using librosa library #####
        signal, _sample_rate = librosa.core.load(filepath, 
                                                sr = self.sample_rate, 
                                                mono = False)

        ##### 2. Change the datatype as float #####
        if signal.dtype == np.int16:
            signal = signal.astype(np.float32)
            signal /= 32768.

        ##### 3. Pad the audio file to match its length with audio_length #####
        if len(signal) < self.audio_length: 
            delta = self.audio_length - len(signal)
            signal = np.pad(signal, (0, delta), mode = 'edge')
        else:
            signal = signal[:self.audio_length]

        ##### 3. Reshape the signal #####
        n_channels = 1 if signal.ndim == 1 else signal.shape[1]        
        signal = signal.reshape(n_channels, signal.shape[0])
            
        ##### 4. Normalize the signal #####
        signal /= np.max(np.abs(signal))

        return signal
    
    
    def batch_generator(self):
        '''
        Generates batches of audios
        '''
        audio_filepath_list = [os.path.join(root, fname)
                              for (root, dir_names, file_names) in os.walk(self.audio_dir, followlinks=True)
                              for fname in file_names
                              if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'))]
        
        audios = []
        for audio_filepath in audio_filepath_list:
            audios.append(self.decode_audio(audio_filepath))
        
        n_batches = len(audios)//self.batch_size

        data = TensorDataset(torch.from_numpy(np.array(audios)))
        dataloader = DataLoader(data, batch_size = self.batch_size, shuffle = True, drop_last = True)
        
        return dataloader, n_batches
