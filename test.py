import os,random,warnings,time,math
import pickle
import numpy as np
import torch
import torch.nn as nn

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse
import pdb
from tqdm import tqdm

from dataloader.data_loader import prepare_dataset, _collate_fn
from dataloader.augment import BackgroundNoise
from base_builder.model_builder import build_model
from dataloader.vocabulary import KsponSpeechVocabulary
from omegaconf import OmegaConf
from metric.metric import *
from metric.loss import *
from torch.utils.data import DataLoader


def test(config):
    # Configuration
    if config.model.use_jaso:
        config.train.transcripts_path_valid = config.train.transcripts_path_valid.replace('.txt','_js.txt')
        config.train.vocab_label = config.train.vocab_label.replace('.csv','_js.csv')
        
    vocab = KsponSpeechVocabulary(config.train.vocab_label)

    model = torch.load(config.model.model_path, map_location=lambda storage, loc: storage).to(device)
    model.eval()
    
    criterion = Attention_Loss(config, vocab)
    metric = get_metric(config, vocab)
    if config.model.use_jaso:
        from copy import deepcopy
        _config = deepcopy(config)
        _config.model.use_jaso = False
        metric_jaso = get_metric(_config, vocab)

    collate_fn = lambda batch: _collate_fn(batch, config)
    validset = prepare_dataset(config, config.train.transcripts_path_valid, vocab, Train=False)
    valid_loader = torch.utils.data.DataLoader(dataset=validset, batch_size=config.train.batch_size,
                                               shuffle=False, collate_fn = collate_fn, 
                                               num_workers=config.train.num_workers)
    
    print(f'validset : {len(validset)}, {len(valid_loader)} batches')
    
    begin_time = time.time()
    print('Test Start')
    
    loss = 0.
    cer = 0.
    
    with torch.no_grad():
        progress_bar = enumerate(valid_loader)
        for i, (video_inputs,audio_inputs,targets,video_input_lengths,audio_input_lengths,target_lengths) in progress_bar:
            video_inputs = video_inputs.to(device)
            video_input_lengths = video_input_lengths.to(device)
            audio_inputs = audio_inputs.to(device)
            audio_input_lengths = audio_input_lengths.to(device)
            targets = targets.to(device)
            target_lengths = torch.as_tensor(target_lengths).to(device)
            
            model_args = [video_inputs, video_input_lengths,\
                          audio_inputs, audio_input_lengths,]
                          
            outputs, output_lengths = model(*model_args)

            # drop final_token from outputs & drop sos_token from targets
            loss_outputs = outputs[:,:-1,:]
            # Except SOS
            loss_target = targets[:, 1:]
            
            batch_loss = criterion(loss_outputs, loss_target, target_lengths).cpu().item()
            print()
            batch_cer = metric(loss_target, loss_outputs, show=True)
            if config.model.use_jaso:
                batch_cer_jaso = metric_jaso(loss_target, loss_outputs, 
                                             target_lengths, output_lengths, show=True)
                print(f'CER : {batch_cer}  /  CER_JASO : {batch_cer_jaso}')
                
            loss += batch_loss
            cer += batch_cer
            
            # progress_bar.set_description(
            #     f'{i+1}/{len(valid_loader)}, Loss:{batch_loss} CER:{batch_cer}'
            # )
            
    print(f'Mean Loss : {loss/(i+1)} Mean CER : {cer/(i+1)}')
    
    
    
def main(config):
    test(config)


    
def get_args():
    parser = argparse.ArgumentParser(description='각종 옵션')
    parser.add_argument('-m','--model',
                        required=True, help='Select Model')
    parser.add_argument('-mp','--model_path',
                        required=False, help='Model Path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = get_args()
    
    config = OmegaConf.load(f"config/test.yaml")

    config.model['model_path'] = args.model_path

    test(config)
