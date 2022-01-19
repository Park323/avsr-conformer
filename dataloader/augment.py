import os
import random
import math

import numpy as np
import tqdm
import torchaudio
from torchaudio import functional as F
from torch import Tensor


class SpecAugment(object):
    def __init__(self, freq_mask_para: int = 18, time_mask_num: int = 10, freq_mask_num: int = 2) :
        self.freq_mask_para = freq_mask_para
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num

    def __call__(self, feature: Tensor) :
        """ Provides SpecAugmentation for audio """
        time_axis_length = feature.size(0)
        freq_axis_length = feature.size(1)
        time_mask_para = time_axis_length / 20      # Refer to "Specaugment on large scale dataset" paper

        # time mask
        for _ in range(self.time_mask_num):
            t = int(np.random.uniform(low=0.0, high=time_mask_para))
            t0 = random.randint(0, time_axis_length - t)
            feature[t0: t0 + t, :] = 0

        # freq mask
        for _ in range(self.freq_mask_num):
            f = int(np.random.uniform(low=0.0, high=self.freq_mask_para))
            f0 = random.randint(0, freq_axis_length - f)
            feature[:, f0: f0 + f] = 0

        return feature

def _get_sample(path, resample=None):
    effects = [["remix", "1"]]
    if resample:
        effects.extend(
            [
                ["lowpass", f"{resample // 2}"],
                ["rate", f"{resample}"],
            ]
        )
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def get_sample(path, resample=None):
    return _get_sample(path, resample=resample)

class BackgroundNoise(object):
    def __init__(self,noise_path:str,sr:int):
        np.random.seed(1)
        random.seed(1)

        self.sr = sr
        self.noise_path = noise_path
        if noise_path[-1] != '/':
            self.noise_path += '/'
        self.noise_list = os.listdir(noise_path)
        self.noise_wav = []
        print("Load Background Noise Data")
        for file in tqdm.tqdm(self.noise_list):
          bg,_ = get_sample(self.noise_path+file,self.sr)
          self.noise_wav.append(bg)
        print("Complete!")
    
    def __call__(self,audio,is_path=True):
        
        idx = np.random.randint(len(self.noise_list))
        noise = self.noise_wav[idx]
        if is_path:
          audio, _ = get_sample(audio,self.sr)
          
        noise_len = noise.shape[1]
        audio_len = audio.shape[1]

        start = np.random.randint(noise_len-audio_len)
        noise = noise[:, start:start+audio.shape[1]]

        audio_power = audio.norm(p=2)
        noise_power = noise.norm(p=2)
        
        snr_db = np.random.randint(low=0,high=20)
        if snr_db == 0:
            return audio
          
        snr = math.exp(snr_db / 10)
        scale = snr * noise_power / audio_power
        noisy_speech = (scale * audio + noise) / 2

        return noisy_speech
        
        

