import os,random,warnings,time,math
import argparse
import torch
import torch.nn as nn
from dataloader.data_loader import prepare_dataset, _collate_fn
from base_builder.model_builder import build_model
from dataloader.vocabulary import KsponSpeechVocabulary
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from metric.metric import CharacterErrorRate
from metric.wer_utils import compute_ctc_uer
from checkpoint.checkpoint import Checkpoint
from torch.utils.data import DataLoader
from itertools import groupby
import pdb
import pandas as pd
import tqdm



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def test(config):

    os.environ["CUDA_VISIBLE_DEVICES"]= config.train.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vocab = KsponSpeechVocabulary(config.train.vocab_label)
    test_model = config.model.model_path+'las_model'+'/'+config.model.exp_day+'/'+config.model.model_file
    #model = build_model(config,vocab)
    model = torch.load(test_model, map_location=lambda storage, loc: storage).to(device)

    model.eval()
    test_metric = CharacterErrorRate(vocab)
    print(model)
    print(count_parameters(model))
    # pdb.set_trace()
    model.eval()  

    testset = prepare_dataset(config, config.train.transcripts_path_test,vocab, Train=False)
    print('preparing end')
    test_loader = torch.utils.data.DataLoader(dataset=testset,batch_size =config.train.batch_size,
                                shuffle=False,collate_fn = _collate_fn, num_workers=config.train.num_workers)
    print('loading end')

    start_time = time.time()
    
    submission = []
    with torch.no_grad():
        for i, (video_inputs,audio_inputs,_,video_input_lengths,audio_input_lengths,_) in tqdm.tqdm(enumerate(test_loader)):
            #video_inputs = video_inputs.to(device)
            audio_inputs = audio_inputs.to(device)

            #video_input_lengths = video_input_lengths.to(device)
            audio_input_lengths = audio_input_lengths.to(device)
            
            model = model
            y_hats = model.greedy_search(audio_inputs, audio_input_lengths)
            # pdb.set_trace()
            for i in range(y_hats.size(0)):
              submission.append(vocab.label_to_string(y_hats[i].cpu().detach().numpy()))
            #print(vocab.label_to_string(predicted))
    print(submission[0])
    print("Total Time")
    print(time.time() - start_time)
    for i in submission:
      print(i)#
    df = pd.read_csv('../sample_submission.csv')
    df['answer'] = submission
    num = 0
    filelist = os.listdir('./submission')
    while f'submission{num}.csv' in filelist:
        num+=1
    df.to_csv('./submission/submission'+str(num)+'.csv',encoding='utf-8-sig',index=False)
    print(f'submission{num} is saved!')
        

if __name__ == '__main__':
    config = OmegaConf.load('test.yaml')
    parser = argparse.ArgumentParser(description='각종 추가 옵션')
    parser.add_argument('--model_file', required=False, default=config.model['model_file'], help='모델이름 입력')
    parser.add_argument('--exp_day',required=False, default=config.model['exp_day'],help='모델 저장경로' )
    args = parser.parse_args()
    config.model.model_file = args.model_file
    config.model.exp_day = args.exp_day

    test(config)
    

