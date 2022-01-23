import torch
import torch.nn as nn
from omegaconf import DictConfig
from dataloader.vocabulary import Vocabulary
from las_model.las import ListenAttendSpell
from model.avsr_conformer import AudioVisualConformer

import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(config, vocab):
    model_name = config.model.name
    if model_name=='las':
        model = build_las_model(config, vocab=vocab)
    elif model_name=='conf':
        model = build_conf_model(config)
        
    print("model parameter ")
    print(count_parameters(model))
    return model

def build_las_model(config, vocab=None, *args, **kargs):
    return ListenAttendSpell(config, vocab) 

def build_conf_model(config, *args, **kargs):
    return AudioVisualConformer(config)