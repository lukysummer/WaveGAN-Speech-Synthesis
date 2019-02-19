import numpy as np
import os

import librosa
import pescador


########################### 1. READ IN AUDIO DATA #############################
class LoadandProcessData():
    
    def __init__(self, audio_dir, batch_size, sample_rate = 16000, audio_length = 16384):
        
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        

    def decode_audio(self, filepath):
        '''
        Reads in an audio file and pre-processes it
        : param filepath: path to the audio file
        : param _sampel_rate: sample rate of the audio file [samples/second]
        '''
        ##### 1. Read in the wavfile using librosa library #####
        signal, _sample_rate = librosa.core.load(filepath, 
                                                sr = self.sample_rate, 
                                                mono = False)
        
        ##### 2. Change the datatype as float #####
        if signal.dtype == np.int16:
            signal = signal.astype(np.float32)
            signal /= 32768.
            
        ##### 3. Adjust the length of the audio file to audio_length #####
        if len(signal) < self.audio_length:
            total_padding = self.audio_length - len(signal)
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            
            signal = np.pad(signal, (left_padding, right_padding), mode = 'constant')
        else:
            signal = signal[:self.audio_length]
                 
        ##### 4. Reshape the signal #####
        n_channels = 1 if signal.ndim == 1 else signal.shape[1]        
        signal = signal.reshape(n_channels, signal.shape[0])
            
        ##### 5. Normalize the signal #####
        signal /= np.max(np.abs(signal))
        
        yield {'X': signal}
    
    
    def batch_generator(self):
        '''
        Generates batches of audios
        : param audio_filepath_list: list of audio file paths
        : param batch_size: desired batch size 
        '''
        audio_filepath_list = [os.path.join(root, fname)
                              for (root, dir_names, file_names) in os.walk(self.audio_dir, followlinks=True)
                              for fname in file_names
                              if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'))]
        
        streamers = []
        for audio_filepath in audio_filepath_list:
            s = pescador.Streamer(self.decode_audio, audio_filepath)
            streamers.append(s)
        
        #print("Number of Training Examples: \n", len(streamers))
        n_batches = len(streamers)//self.batch_size
        mux = pescador.ShuffledMux(streamers)
        batch_gen = pescador.buffer_stream(mux, self.batch_size)
    
        return batch_gen, n_batches
