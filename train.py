import os,random,warnings,time,math
import pickle
import numpy as np
import torch
import torch.nn as nn

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse
import pdb
from tqdm import tqdm

from GPUtil import showUtilization as gpu_usage
from dataloader.data_loader import prepare_dataset, _collate_fn
from dataloader.augment import BackgroundNoise
from base_builder.model_builder import build_model
from dataloader.vocabulary import KsponSpeechVocabulary
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from metric.metric import *
from metric.loss import *
from checkpoint.checkpoint import Checkpoint
from torch.utils.data import DataLoader
from scheduler import CosineAnnealingWarmUpRestarts, NoamLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 작업 그룹 초기화
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_mp(fn, world_size, *args):
    mp.spawn(fn,
             args = (world_size, *args),
             nprocs=world_size,
             join=True)
    
    
    
def _train(config, model, dataloader, optimizer, scheduler, criterion, metric, vocab,
                    train_begin_time, epoch, summary, device='cuda'):
    log_format = "epoch: {:4d}/{:4d}, step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    cers = []
    epoch_loss_total = 0.
    total_num = 0
    timestep = 0
    begin_time = epoch_begin_time = time.time() #모델 학습 시작 시간

    model.train()
    progress_bar = tqdm(enumerate(dataloader),ncols=110)
    for i, (video_inputs,audio_inputs,targets,video_input_lengths,audio_input_lengths,target_lengths) in progress_bar:
        optimizer.zero_grad()

        video_inputs = video_inputs.to(device)
        video_input_lengths = video_input_lengths.to(device)

        audio_inputs = audio_inputs.to(device)
        audio_input_lengths = audio_input_lengths.to(device)

        targets = targets.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        model = model
        
        model_args = [video_inputs, video_input_lengths,\
                      audio_inputs, audio_input_lengths,\
                      targets, target_lengths]
        
        outputs = model(*model_args)
            
        loss_target = targets[:, 1:]
        if config.model.name != 'las':
            if config.decoder.method=='att_only':
                loss_outputs = outputs[:,:-1]
                metric_outputs = outputs
            elif config.decoder.method=='ctc_only':
                loss_outputs = outputs
                metric_outputs = outputs
            else:
                loss_outputs = (outputs[0][:,:-1], outputs[1])
                metric_outputs = outputs[0]
        else:
            loss_outputs = outputs
            metric_outputs = outputs
        loss = criterion(loss_outputs, loss_target, target_lengths)
        
        cer = metric(loss_target, metric_outputs, show=False)
        cers.append(cer) # add cer on this epoch
        
        loss.backward()
        optimizer.step()

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
        
        scheduler.step()
    
    summary.add_scalar('learning_rate/lr',optimizer.state_dict()['param_groups'][0]['lr'],epoch)
    train_loss, train_cer = epoch_loss_total / total_num, sum(cers) / len(cers)
    
    return train_loss, train_cer



#def train(rank, world_size, config):
def train(config):
    # multi processing
    #print(f"Running DDP training on rank {rank}.")
    #setup(rank, world_size)
    
    # 시드 고정
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)
    
    # Configuration
    if config.model.use_jaso:
        config.train.transcripts_path_train = config.train.transcripts_path_train.replace('.txt','_js.txt')
        config.train.transcripts_path_valid = config.train.transcripts_path_valid.replace('.txt','_js.txt')
        config.train.vocab_label = config.train.vocab_label.replace('.csv','_js.csv')
        
    vocab = KsponSpeechVocabulary(config.train.vocab_label)

    if not config.train.resume: # 학습한 경우가 없으면,
        model = build_model(config, vocab)
        start_epoch =0
        if config.train.multi_gpu == True:
            model = nn.DataParallel(model)
        model = model.to(device)
#        model = model.to(rank)
#        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-13)
        scheduler = NoamLR(optimizer, 
                          [2.5], 
                          [config.train.num_epochs],
                          [10000],
                          [1e-13],
                          [config.train.learning_rate],
                          [1e-7])
#        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=config.train.learning_rate, T_up=3, gamma=0.5)
#        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
#        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
   
    else: # 학습한 경우가 있으면,
#        model = model.to(rank)
#        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        checkpoint = Checkpoint(config=config)
        latest_checkpoint_path = checkpoint.get_latest_checkpoint()
        resume_checkpoint = checkpoint.load(latest_checkpoint_path) ##N번째 epoch부터 학습하기
        model = resume_checkpoint.model
        optimizer = resume_checkpoint.optimizer
        scheduler = resume_checkpoint.scheduler
        start_epoch = resume_checkpoint.epoch        
        if isinstance(model, nn.DataParallel):
            model = model.module
        if config.train.multi_gpu == True:
#            model = model.to(device)
#            model = model.to(rank)
#            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
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
    
#    sampler = DistributedSampler(trainset, world_size, rank)
#    mp_context = torch.multiprocessing.get_context('fork')
#    collate_fn = lambda batch: _collate_fn(batch, config)
#    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.train.batch_size,
#                                               collate_fn = collate_fn, 
#                                               multiprocessing_context=mp_context,
#                                               sampler=sampler,
#                                               num_workers=config.train.num_workers)
    collate_fn = lambda batch: _collate_fn(batch, config)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.train.batch_size,
                                               collate_fn = collate_fn, shuffle=True,
                                               num_workers=config.train.num_workers)
    
    print(f'trainset : {len(trainset)}, {len(train_loader)} batches')
    
    train_begin_time = time.time()
    print("Initial GPU Usage")  
    gpu_usage()
    print('Train start')
    for epoch in range(start_epoch, config.train.num_epochs):
        #dist.barrier()
        #train_loss, train_cer = _train(config, model, train_loader, 
        #                              optimizer, criterion, train_metric, vocab,
        #                              train_begin_time, epoch, summary, device=rank)
        train_loss, train_cer = _train(config, model, train_loader, 
                                      optimizer, scheduler, criterion, train_metric, vocab,
                                      train_begin_time, epoch, summary, device=device)
        
        tr_sentences = 'Epoch %d Training Loss %0.4f CER %0.5f '% (epoch+1, train_loss, train_cer)
        
        summary.add_scalar('training/loss',train_loss,epoch)
        summary.add_scalar('training/cer',train_cer,epoch)

        print(tr_sentences)
        train_metric.reset()
        
#        if config.train.scheduler=='on':
#            scheduler.step(train_loss)
        #if rank == 0:
        Checkpoint(model, optimizer, scheduler, epoch+1, config=config).save()
        
    #cleanup()



def test(config):
    # Configuration
    ## 자소
    if config.model.use_jaso:
        config.train.transcripts_path_train = config.train.transcripts_path_train.replace('.txt','_js.txt')
        config.train.transcripts_path_valid = config.train.transcripts_path_valid.replace('.txt','_js.txt')
        config.train.vocab_label = config.train.vocab_label.replace('.csv','_js.csv')
    
    vocab = KsponSpeechVocabulary(config.train.vocab_label)

    model = torch.load(config.model.model_path, map_location=lambda storage, loc: storage).to(device)
    #model.config.model.max_len = config.model.max_len # 
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
    cer_js = 0.
    with torch.no_grad():
        # progress_bar = tqdm(enumerate(valid_loader),ncols=110)
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
            metric.reset()
            if config.model.use_jaso:
                batch_cer_jaso = metric_jaso(loss_target, loss_outputs, 
                                             target_lengths, output_lengths, show=True)
                print(f'CER : {batch_cer}  /  CER_JASO : {batch_cer_jaso}')
                cer_js += batch_cer_jaso
                metric_jaso.reset()
                
            loss += batch_loss
            cer += batch_cer
            
            # progress_bar.set_description(
            #     f'{i+1}/{len(valid_loader)}, Loss:{batch_loss} CER:{batch_cer}'
            # )
    if config.model.use_jaso:
        print(f'Mean Loss : {loss/(i+1)} Mean CER : {cer/(i+1)}')
    else:
        print(f'Mean Loss : {loss/(i+1)} Mean CER : {cer/(i+1)} Mean CER JASO : {cer_js/(i+1)}')



def main(config):
    print("cuda : ", torch.cuda.is_available())
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    
    #world_size = 2
    #run_mp(train, world_size, config)
    train(config)
    


def get_args():
    parser = argparse.ArgumentParser(description='각종 옵션')
    parser.add_argument('-r','--resume',
                        action='store_true', help='RESUME - 이어서 학습하면 True')
    parser.add_argument('-s','--sample',
                        action='store_true', help='USE_ONLY_SAMPLE_FOR_DEBUGGING')
    parser.add_argument('-m','--model',
                        required=True, help='Select Model')
    parser.add_argument('--mode', default='train', help='Select Mode')
    parser.add_argument('-mp', '--model_path',
                        required=False, help='Model Path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_args()
    
    config = OmegaConf.load(f"config/{args.model}.yaml")

    config.train['resume'] = args.resume
    config.model['model_path'] = args.model_path
    if args.sample:
        config.train.transcripts_path_train = 'dataset/Sample.txt'
        config.train.transcripts_path_valid = 'dataset/Sample.txt'

    if args.mode == 'train':
        main(config)
    else :
        test(config)

