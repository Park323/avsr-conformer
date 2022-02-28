import os,random,warnings,time,math
import pickle
import argparse
import pdb
from tqdm import tqdm
from GPUtil import showUtilization as gpu_usage

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader.data_loader import prepare_dataset, _collate_fn
from dataloader.augment import BackgroundNoise
from base_builder.model_builder import build_model
from dataloader.vocabulary import KsponSpeechVocabulary
from omegaconf import OmegaConf
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from metric.metric import *
from metric.loss import *
from checkpoint.checkpoint import Checkpoint

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
print("cuda : ", torch.cuda.is_available())
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())


def train_on_epoch(config, model, dataloader, optimizer, scheduler, criterion, metric, vocab,
                   train_begin_time, epoch, summary, device='cuda', train=True):
    log_format = "epoch: {:4d}/{:4d}, step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    cers = []
    epoch_loss_total = 0.
    max_grad_norm = 5.
    total_num = 0
    timestep = 0
    begin_time = epoch_begin_time = time.time()
    
    progress_bar = tqdm(enumerate(dataloader),ncols=110)
    for i, (video_inputs,audio_inputs,targets,video_input_lengths,audio_input_lengths,target_lengths) in progress_bar:
        
        video_inputs = video_inputs.to(device)
        video_input_lengths = video_input_lengths.to(device)
        audio_inputs = audio_inputs.to(device)
        audio_input_lengths = audio_input_lengths.to(device)

        targets = targets.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        
        model_args = [video_inputs, video_input_lengths,\
                      audio_inputs, audio_input_lengths,\
                      targets, target_lengths]
                      
        with torch.cuda.amp.autocast():
            outputs, output_lengths = model(*model_args)
        
        # drop final_token from outputs & drop sos_token from targets
        loss_outputs = (outputs[0][:,:-1,:], outputs[1]) if isinstance(outputs, tuple) else outputs[:,:-1,:]
        # Except SOS
        loss_target = targets[:, 1:]
        
        loss = criterion(loss_outputs, loss_target, target_lengths)
        cer = metric(loss_outputs[0], output_lengths, loss_target, target_lengths, show=False)
        cers.append(cer) # add cer on this epoch
        
        loss.backward()
        #print("####GRAD#####")
        #print(model.module.ceLinear.weight.grad)
        optimizer.step()
        optimizer.zero_grad()
        
        total_num += 1
        epoch_loss_total += loss.item()

        timestep += 1
        
        if timestep % config.train.print_every == 0:
            current_time = time.time()
            elapsed = current_time - begin_time
            epoch_elapsed = (current_time - epoch_begin_time) / 60.0
            train_elapsed = (current_time - train_begin_time) / 3600.0
            
            progress_bar.set_description(
                log_format.format(epoch+1,config.train.num_epochs,
                timestep, len(dataloader), loss,
                cer, elapsed, epoch_elapsed, train_elapsed,
                optimizer.state_dict()['param_groups'][0]['lr'])
            )
            begin_time = time.time()
            
        summary.add_scalar('iter_training/loss',loss,epoch*len(dataloader)+i)
        summary.add_scalar('iter_training/cer',cer,epoch*len(dataloader)+i)
    
    summary.add_scalar('learning_rate/lr',optimizer.state_dict()['param_groups'][0]['lr'],epoch)
    train_loss, train_cer = epoch_loss_total / total_num, sum(cers) / len(cers)

    scheduler.step(train_loss)

    return train_loss, train_cer


def train(config):

    vocab = KsponSpeechVocabulary(config.train.vocab_label)

    if not config.train.resume: 
        model = build_model(config, vocab)
        start_epoch = 0
        if config.train.multi_gpu == True:
            model = nn.DataParallel(model)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
   
    else: 
        checkpoint = Checkpoint(config=config)
        latest_checkpoint_path = checkpoint.get_latest_checkpoint()
        resume_checkpoint = checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model
        optimizer = resume_checkpoint.optimizer
        scheduler = resume_checkpoint.scheduler
        start_epoch = resume_checkpoint.epoch
        if isinstance(model, nn.DataParallel):
            model = model.module
        if config.train.multi_gpu == True:
            model = nn.DataParallel(model)
        model = model.to(device)
        print(f'Loaded train logs, start from {resume_checkpoint.epoch+1} epoch')

    criterion = get_criterion(config, vocab)
    train_metric = get_metric(config, vocab)

    tensorboard_path = f'outputs/tensorboard/{config.model.name}/{config.train.exp_day}'
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    summary = SummaryWriter(tensorboard_path)

    trainset = prepare_dataset(config, config.train.transcripts_path_train, vocab, Train=True)
    
    collate_fn = lambda batch: _collate_fn(batch, config, pad_id=vocab.pad_id)
    train_loader_0 = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.train.batch_size,
                                                 shuffle=False, collate_fn = collate_fn, 
                                                 num_workers=config.train.num_workers)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.train.batch_size,
                                               shuffle=True, collate_fn = collate_fn, 
                                               num_workers=config.train.num_workers)
    
    print(f'trainset : {len(trainset)}, {len(train_loader)} batches')
    
    model.train()
    print('Train start')
    train_begin_time = time.time()
    for epoch in range(start_epoch, config.train.num_epochs):
        data = train_loader if epoch else train_loader_0
        
        train_loss, train_cer = train_on_epoch(config, model, train_loader, 
                                               optimizer, scheduler, criterion, train_metric, vocab,
                                               train_begin_time, epoch, summary, device)
        
        tr_sentences = 'Epoch %d Training Loss %0.4f CER %0.5f '% (epoch+1, train_loss, train_cer)
        
        summary.add_scalar('training/loss',train_loss,epoch)
        summary.add_scalar('training/cer',train_cer,epoch)

        print(tr_sentences)
        train_metric.reset()
        
        Checkpoint(model, optimizer, scheduler, epoch+1, config=config).save()


def test(config):
    
    vocab = KsponSpeechVocabulary(config.train.vocab_label)

    model = torch.load(config.model.model_path, map_location=lambda storage, loc: storage).to(device)
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.config.model.max_len = config.model.max_len 
    # module.config.decoder.method = 'att_only'#'hybrid'
    # module.config.decoder.ctc_att_rate = 0.
    model.eval()
    
    criterion = Attention_Loss(config, vocab)
    metric = get_metric(config, vocab)

    collate_fn = lambda batch: _collate_fn(batch, config, pad_id=vocab.pad_id)
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
        progress_bar = tqdm(enumerate(valid_loader),ncols=110)
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
#            for i in range(y_hats.size(0)):
#                submission.append(vocab.label_to_string(y_hats[i].cpu().detach().numpy()))

            # drop final_token from outputs & drop sos_token from targets
            loss_outputs = outputs[:,:-1,:]
            # Except SOS
            loss_target = targets[:, 1:]
            
            batch_loss = criterion(loss_outputs, loss_target, target_lengths).cpu().item()
            batch_cer =metric(loss_outputs, output_lengths, loss_target, target_lengths, show=True) 
            loss += batch_loss
            cer += batch_cer
            
            progress_bar.set_description(
                f'{i+1}/{len(valid_loader)}, Loss:{batch_loss} CER:{batch_cer}'
            )
            
    print(f'Mean Loss : {loss/(i+1)} Mean CER : {cer/(i+1)}')

def main(config):
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)
    
    if config.train.mode=='train':
        train(config)
    else:
        test(config)
        
    
        
def get_args():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('-r','--resume',
                        action='store_true', help='RESUME - True')
    parser.add_argument('-s','--sample',
                        action='store_true', help='USE_ONLY_SAMPLE_FOR_DEBUGGING')
    parser.add_argument('-m','--model',
                        required=True, help='Select Model')
    parser.add_argument('--mode', default='train', help='Select Mode')
    parser.add_argument('--model_path',
                        required=False, help='Model Path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_args()
    
    config = OmegaConf.load(f"config/{args.model}.yaml")

    config.train['resume'] = args.resume
    config.train['mode'] = args.mode
    config.model['model_path'] = args.model_path
    if args.sample:
        config.train.transcripts_path_train = 'dataset/Sample.txt'
        config.train.transcripts_path_valid = 'dataset/Sample.txt'

    main(config)

