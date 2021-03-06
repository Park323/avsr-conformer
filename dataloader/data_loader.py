import os
import math
import torch
import random

from omegaconf import DictConfig
from torch.utils.data import Dataset
from dataloader.vocabulary import Vocabulary
from dataloader.augment import SpecAugment, BackgroundNoise, get_sample
from dataloader.feature import MelSpectrogram,MFCC,Spectrogram,FilterBank
from torch import Tensor, FloatTensor
import torchaudio
import numpy as np
import pdb
import csv
import sys
from numpy.lib.stride_tricks import as_strided
import warnings
#import librosa

def load_dataset(transcripts_path):
    """
    Provides dictionary of filename and labels
    Args:
        transcripts_path (str): path of transcripts
    Returns: target_dict
        - **target_dict** (dict): dictionary of filename and labels
    """
    video_paths = list()
    audio_paths = list()
    korean_transcripts = list()
    transcripts = list()

    with open(transcripts_path) as f:
        for idx, line in enumerate(f.readlines()):
            # pdb.set_trace()
            video_path, audio_path, korean_transcript, transcript = line.split('\t')
            transcript = transcript.replace('\n', '')
            video_paths.append(video_path)
            audio_paths.append(audio_path)
            korean_transcripts.append(korean_transcript)
            transcripts.append(transcript)

    return video_paths, audio_paths, korean_transcripts, transcripts

def prepare_dataset(config, transcripts_path: str, vocab: Vocabulary, Train=True):

    train_or_test = 'train' if Train else 'test'
    print(f"prepare {train_or_test} dataset start !!")

    tr_video_paths, tr_audio_paths, tr_korean_transcripts, tr_transcripts = load_dataset(transcripts_path)

    if Train == True:
        trainset = AV_Dataset(
                video_paths=tr_video_paths,
                audio_paths=tr_audio_paths,
                korean_transcripts=tr_korean_transcripts,
                transcripts=tr_transcripts,
                sos_id=vocab.sos_id, 
                eos_id=vocab.eos_id,
                config=config,
                noise_augment=True
                )
    elif Train == False:
        trainset = AV_Dataset(
            video_paths=tr_video_paths,
            audio_paths=tr_audio_paths,
            korean_transcripts=tr_korean_transcripts,
            transcripts=tr_transcripts,
            sos_id=vocab.sos_id, 
            eos_id=vocab.eos_id,
            config=config,
            )
    
    print(f"prepare {train_or_test} dataset finished.")

    return trainset



class AV_Dataset(Dataset):
    
    
    def __init__(
            self,
            video_paths: list,              # list of video paths
            audio_paths: list,              # list of audio paths
            korean_transcripts: list,
            transcripts: list,              # list of transcript paths
            sos_id: int,                    # identification of start of sequence token
            eos_id: int,                    # identification of end of sequence token
            config,             # set of arguments
            spec_augment: bool = False,     # flag indication whether to use spec-augmentation of not
            noise_augment: bool = False,     # flag indication whether to use spec-augmentation of not
            ):
        super(AV_Dataset, self).__init__()
        
        self.config = config
        
        if config.audio.transform_method.lower() == 'fbank':
            self.filterbank = FilterBank(config.audio.sample_rate, 
                                config.audio.n_mels, 
                                config.audio.frame_length, 
                                config.audio.frame_shift,
                                )
            def filterbank(x):
                x = self.filterbank(x)
                x = np.transpose(x, (1,0))
                return x
            self.transforms = filterbank
        elif config.audio.transform_method.lower() == 'raw':
            self.transforms = lambda x: np.expand_dims(x,1)
            
        self.video_paths = list(video_paths)
        self.audio_paths = list(audio_paths)
        self.korean_transcripts = list(korean_transcripts)
        self.transcripts = list(transcripts)
        self.dataset_size = len(self.audio_paths)

        self.sos_id=sos_id
        self.eos_id=eos_id
        self.normalize = config.audio.normalize

        self.VANILLA = 0           # Not apply augmentation
        self.SPEC_AUGMENT = 1      # 1 : SpecAugment, 2: NoiseAugment, 3: Both
        self.NOISE_AUGMENT = 2
        self.BOTH_AUGMENT = 3
        self.augment_methods = [self.VANILLA] * len(self.audio_paths)
        #self.augment_methods = np.random.choice([0,1,2,3], len(self.audio_paths), p=[0.6, 0.15, 0.15, 0.1])
        
        self.spec_augment = SpecAugment(config.audio.freq_mask_para, 
                                    config.audio.time_mask_num, 
                                    config.audio.freq_mask_num,
                                    )
        self.noise_augment = BackgroundNoise(config.train.noise_path, 
                                            self.config.audio.sample_rate)

        self._augment(spec_augment, noise_augment)

    def __getitem__(self, index):
        if self.config.video.use_vid:
            video_feature = self.parse_video(self.video_paths[index])
        else:
            # return dummy video feature
            video_feature = torch.Tensor([[0]])
        audio_feature = self.parse_audio(self.audio_paths[index],self.augment_methods[index])
        transcript = self.parse_transcript(self.transcripts[index])
        korean_transcript = self.parse_korean_transcripts(self.korean_transcripts[index])
        return video_feature, audio_feature, transcript, korean_transcript,
    
    def parse_audio(self,audio_path: str, augment_method):
        # pdb.set_trace()
        signal, _ = get_sample(audio_path,resample=self.config.audio.sample_rate)
        if augment_method in [self.NOISE_AUGMENT, self.BOTH_AUGMENT]:
            signal = self.noise_augment(signal, is_path=False)
        signal = signal.numpy().reshape(-1,)    
        feature = self.transforms(signal)
        if self.normalize:
            feature -= feature.mean()
            feature /= np.std(feature)

        feature = FloatTensor(feature)

        if augment_method in [self.SPEC_AUGMENT, self.BOTH_AUGMENT]:
            feature = self.spec_augment(feature)
            
        return feature
    
    def parse_video(self, video_path: str):
        if self.config.video.use_npz:
            if 'lip' in self.config.train.transcripts_path_train:
                video = np.load(video_path)['data']
            else:
                video = np.load(video_path)['video']
                video = video.transpose(1,0)
        else:
            video = np.load(video_path)
            if self.config.video.use_raw_vid != 'on':
                video = video.transpose(1,0)
        video = torch.from_numpy(video).float()
        video -= torch.mean(video)
        video /= torch.std(video)
        video_feature = video
        return video_feature

    def parse_transcript(self, transcript):
        tokens = transcript.split(' ')
        transcript = list()

        transcript.append(int(self.sos_id))
        for token in tokens:
            transcript.append(int(token))
        transcript.append(int(self.eos_id))

        return transcript
    
    def parse_korean_transcripts(self, korean_transcript):
        
        tokens = korean_transcript.split(' ')
        korean_transcript = list()

        korean_transcript.append(str(self.sos_id))
        for token in tokens:
            korean_transcript.append(str(token))
        korean_transcript.append(str(self.eos_id))

        return korean_transcript

    def _augment(self, spec_augment, noise_augment):
        """ Spec Augmentation """
        if not spec_augment and not noise_augment:
            available_augment = None
        elif not noise_augment:
            available_augment = [self.SPEC_AUGMENT]
        elif not spec_augment:
            available_augment = [self.NOISE_AUGMENT]
        else :
            available_augment = [self.SPEC_AUGMENT,self.NOISE_AUGMENT,self.BOTH_AUGMENT]
        if available_augment:
            print(f"Applying Augmentation...{self.dataset_size}")
            for idx in range(self.dataset_size):
                self.augment_methods.append(np.random.choice(available_augment))
                self.video_paths.append(self.video_paths[idx])
                self.audio_paths.append(self.audio_paths[idx])
                self.korean_transcripts.append(self.korean_transcripts[idx])
                self.transcripts.append(self.transcripts[idx])

            print(f"Augmentation Finished...{len(self.audio_paths)}")

    def shuffle(self):
        """ Shuffle dataset """
        print('shuffle dataset')
        tmp = list(zip(self.video_paths,self.audio_paths,self.korean_transcripts, self.transcripts, self.augment_methods))
        random.shuffle(tmp)
        self.video_paths,self.audio_paths, self.korean_transcripts, self.transcripts, self.augment_methods = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)


def _collate_fn(batch, config):
    """ functions that pad to the maximum sequence length """
    def vid_length_(p):
        return len(p[0])

    def seq_length_(p):
        return len(p[1])

    def target_length_(p):
        return len(p[2])
    
    # sort by sequence length for rnn.pack_padded_sequence()
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    seq_lengths = [len(s[1]) for s in batch]
    target_lengths = [len(s[2]) - 1 for s in batch]
    
    max_seq_sample = max(batch, key=seq_length_)[1]
    # max_target_sample = max(batch, key=target_length_)[2]
    
    max_seq_size = max_seq_sample.size(0)
    # max_target_size = len(max_target_sample)
    max_target_size = config.model.max_len
    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)
    
    seqs = torch.zeros(batch_size, max_seq_size, feat_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(0)
    
    
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[1]
        target = sample[2]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
    
    seq_lengths = torch.IntTensor(seq_lengths)
    
    # B T C  --> B C T
    seqs = seqs.permute(0,2,1)
    
    
    if config.video.use_vid :
        raw = config.video.use_raw_vid == 'on'
        vid_lengths = [len(s[0]) for s in batch]
        max_vid_sample = max(batch, key=vid_length_)[0]
        max_vid_size = max_vid_sample.size(0)
        
        if raw:
            vid_feat_x = max_vid_sample.size(1)
            vid_feat_y = max_vid_sample.size(2)
            vid_feat_c = max_vid_sample.size(3)
            vids = torch.zeros(batch_size, max_vid_size, vid_feat_x,vid_feat_y,vid_feat_c)
        else:
            vid_feat_c = max_vid_sample.size(1)
            vids = torch.zeros(batch_size, max_vid_size, vid_feat_c)
        
        for x in range(batch_size):
            sample = batch[x]
            video_ = sample[0]
            vid_length = video_.size(0)
            if raw:
                vids[x,:vid_length,:,:,:] = video_
            else:
                vids[x,:vid_length,:] = video_
    
        vid_lengths = torch.IntTensor(vid_lengths)
        
        if raw:
            # B T W H C --> B C T W H
            # pdb.set_trace()
            vids = vids.permute(0,4,1,2,3)

    else:
        vids = torch.zeros((batch_size, 1))
        vid_lengths = torch.zeros((batch_size,)).to(int)
    
    
    return vids, seqs, targets, vid_lengths, seq_lengths, target_lengths



