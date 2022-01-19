import torch
import torch.nn as nn
from omegaconf import DictConfig
from astropy.modeling import ParameterError
from dataloader.vocabulary import Vocabulary
from las_model.las import ListenAttendSpell

import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(config, vocab):
    
    model = build_las_model(config, vocab=vocab)
        
    print("model parameter ")
    print(count_parameters(model))
    return model

def build_las_model(opt, vocab):
    return ListenAttendSpell(opt, vocab)