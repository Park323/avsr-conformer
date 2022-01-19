import os,random,warnings,time,math
import pickle
import numpy as np
import torch
import torch.nn as nn
import argparse
import pdb
from tqdm import tqdm

from dataloader.data_loader import prepare_dataset, _collate_fn
from dataloader.augment import BackgroundNoise
from base_builder.model_builder import build_model
from dataloader.vocabulary import KsponSpeechVocabulary
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from metric.metric import CharacterErrorRate
from las_model.layers import LabelSmoothingLoss
from checkpoint.checkpoint import Checkpoint
from torch.utils.data import DataLoader

def train(config):
    #시드 고정
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"]= config.train.gpu #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("cuda : ", torch.cuda.is_available())
    vocab = KsponSpeechVocabulary(config.train.vocab_label)

    if not config.train.resume: # 학습한 경우가 없으면,
        model = build_model(config, vocab)
        start_epoch =0
        if config.train.multi_gpu == True:
            model = nn.DataParallel(model)
        model = model.to(device)
   
    else: # 학습한 경우가 있으면,
        checkpoint = Checkpoint(config=config)
        latest_checkpoint_path = checkpoint.get_latest_checkpoint()
        resume_checkpoint = checkpoint.load(latest_checkpoint_path) ##N번째 epoch부터 학습하기
        model = resume_checkpoint.model
        optimizer = resume_checkpoint.optimizer
        start_epoch = resume_checkpoint.epoch
        if isinstance(model, nn.DataParallel):
            model = model.module
        if config.train.multi_gpu == True:
            model = nn.DataParallel(model)
        model = model.to(device)    
        print(f'Loaded train logs, start from {resume_checkpoint.epoch+1} epoch')
        
    print(model)

    train_metric = CharacterErrorRate(vocab)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    criterion = nn.CTCLoss(blank=vocab.unk_id, reduction='sum', zero_infinity=True)
    
    tensorboard_path = 'outputs/tensorboard/las_model/'+str(config.train.exp_day)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    summary = SummaryWriter(tensorboard_path)

    bg_sound = BackgroundNoise(noise_path=config.train.noise_path,sr=config.audio.sample_rate)

    trainset = prepare_dataset(config, config.train.transcripts_path_train, vocab, Train=True, bg_sound=bg_sound)
    
    print(len(trainset))
    train_loader = torch.utils.data.DataLoader(dataset=trainset,batch_size =config.train.batch_size,
                                shuffle=True,collate_fn = _collate_fn, num_workers=config.train.num_workers)
    
    train_begin_time = time.time()
    print('Train start')
    for epoch in range(start_epoch, config.train.num_epochs):
        #######################################           Train             ###############################################
        train_loss, train_cer = train_on_epoch(config, model, train_loader, optimizer, criterion, train_metric, vocab,
                                         train_begin_time, epoch, summary, device)
        
        tr_sentences = 'Epoch %d Training Loss %0.4f CER %0.5f '% (epoch+1, train_loss, train_cer)
        
        summary.add_scalar('training/loss',train_loss,epoch)
        summary.add_scalar('training/cer',train_cer,epoch)

        Checkpoint(model, optimizer, epoch+1, config=config).save()

        print(tr_sentences)
        train_metric.reset()
        
def train_on_epoch(config, model, dataloader, optimizer, criterion, metric, vocab,
                    train_begin_time, epoch, summary, device='cuda'):
    log_format = "epoch: {:4d}/{:4d}, step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    cers = []
    epoch_loss_total = 0.
    total_num = 0
    timestep = 0
    model.train()
    begin_time = epoch_begin_time = time.time() #모델 학습 시작 시간

    progress_bar = tqdm(enumerate(dataloader),ncols=110)
    for i, (video_inputs,audio_inputs,targets,video_input_lengths,audio_input_lengths,target_lengths) in progress_bar:
        
        optimizer.zero_grad()

        audio_inputs = audio_inputs.to(device)
        audio_input_lengths = audio_input_lengths.to(device)

        targets = targets.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        model = model
        
        model_args = [video_inputs, video_input_lengths,\
                      audio_inputs, audio_input_lengths,\
                      targets, target_lengths]
        
        with torch.cuda.amp.autocast():
            outputs = model(*model_args)
            
        logits = outputs.contiguous().view(-1, outputs.size(-1))
        loss_target = targets[:, 1:].contiguous().view(-1)
        #loss = F.cross_entropy(yhat, loss_target)
        criterion = LabelSmoothingLoss(len(vocab), ignore_index=vocab.pad_id, \
                                        smoothing=config.las_model['label_smoothing']).cuda()
        loss = criterion(logits, loss_target)
            
        y_hats = outputs.max(-1)[1]
        cer = metric(targets[:, 1:], y_hats)
        cers.append(cer) # add cer on this epoch
        loss.backward()
        optimizer.step()

        total_num += int(audio_input_lengths.sum())  
        epoch_loss_total += loss.item()

        timestep += 1
        torch.cuda.empty_cache()
        
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
    return train_loss, train_cer

def get_args(config):
    parser = argparse.ArgumentParser(description='각종 옵션')
    parser.add_argument('-e','--epoch', required=False, default=config.train['num_epochs'], 
                        type=int, help='Epochs 입력')
    parser.add_argument('-lr','--learning_rate',required=False, default=config.train['learning_rate'], 
                        type=float, help='Learning_rate' )
    parser.add_argument('-r','--resume',
                        action='store_true', help='RESUME - 이어서 학습하면 True')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = OmegaConf.load('train.yaml')
    
    args = get_args(config)

    config.train['num_epochs'] = args.epoch
    config.train['resume'] = args.resume
    config.train['learning_rate'] = args.learning_rate

    train(config)

