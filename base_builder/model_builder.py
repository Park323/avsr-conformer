import torch
import torch.nn as nn
from omegaconf import DictConfig
from dataloader.vocabulary import Vocabulary
from las_model.las import ListenAttendSpell
from model.avsr_conformer import AudioVisualConformer, AudioConformer, VideoConformer

import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(config, vocab):
    model_name = config.model.name
    if model_name=='las':
        model = build_las_model(config, vocab=vocab)
    elif model_name in ['conf','trans']:
        model = build_conf_model(config, 'multi', vocab=vocab)
    elif model_name in ['conf_a','trans_a']:
        model = build_conf_model(config, 'audio', vocab=vocab)
    elif model_name in ['conf_v','trans_v']:
        model = build_conf_model(config, 'video', vocab=vocab)
        
    print("model parameter ")
    print(count_parameters(model))
    return model

def build_las_model(config, vocab=None, *args, **kargs):
    return ListenAttendSpell(config, vocab) 

def build_conf_model(config, mode, vocab=None, *args, **kargs):
    if mode=='multi':
        model = AudioVisualConformer(config, vocab)
    elif mode=='audio':
        model = AudioConformer(config, vocab)
    elif mode=='video':
        print('VideoConformer')
        model = VideoConformer(config, vocab)
    return model