import os
import re
import random
from tqdm import tqdm
import glob
import pandas as pd
import numpy as np
import pdb
import argparse

from dataloader.vocabulary import convert_to_JASO

def load_label(filepath, encoding='utf-8'):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding=encoding)
    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    #freq_list = ch_labels.get("freq", id_list)
    #ord_list = ch_labels["ord"]

    for (id_, char) in zip(id_list, char_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()
    
    for ch in sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        except KeyError:
            print(f"KeyError Occured, Key : '{ch}'")
            continue

    return target[:-1]



def target_to_sentence(target, id2char):
    sentence = ""
    targets = target.split()

    for n in targets:
        sentence += id2char[int(n)]
    return sentence



def generate_character_script(videos_paths, audios_paths, transcripts, test=False, valid_rate=0.2):
    print('create_script started..')
    mode = 'Test' if test else 'Train'
    if mode == 'Train':
        char2id, id2char = load_label("./dataset/labels.csv")
        char2id_jaso, _ = load_label("./dataset/labels_js.csv")
    
        
        ####### Train/Valid Split #######
        val_num = int(valid_rate*len(transcripts))
        
        # Check Repeatancy
        redundant_indices = {}
        for i in range(len(transcripts)):
            if transcripts[i] in redundant_indices.keys():
                redundant_indices[transcripts[i]].add(i)
            else:
                redundant_indices[transcripts[i]] = set([i])

        train_videos_paths, train_audios_paths, train_transcripts = list(), list(), list()
        valid_videos_paths, valid_audios_paths, valid_transcripts = list(), list(), list()
        
        while len(valid_transcripts) < val_num:
            sntn = np.random.choice(list(redundant_indices.keys()))
            indices = redundant_indices.pop(sntn)
            for idx in indices:
                valid_videos_paths.append(videos_paths[idx])
                valid_audios_paths.append(audios_paths[idx])
                valid_transcripts.append(transcripts[idx])
        print(f'val num : {len(valid_transcripts)}')
        
        while redundant_indices:
            sntn = list(redundant_indices.keys())[0]
            indices = redundant_indices.pop(sntn)
            for idx in indices:
                train_videos_paths.append(videos_paths[idx])
                train_audios_paths.append(audios_paths[idx])
                train_transcripts.append(transcripts[idx])
        print(f'train num : {len(train_transcripts)}')
                
        trainsets = list(zip(train_videos_paths, train_audios_paths, train_transcripts))
        valsets = list(zip(valid_videos_paths, valid_audios_paths, valid_transcripts))
        
#        # Simple mEthod
#        dataset = list(zip(videos_paths, audios_paths, transcripts))
#        random.shuffle(dataset)
#        trainsets = dataset[:-val_num]
#        valsets = dataset[-val_num:]

        ####### Split End #########

        ### Sort Train Set by the length of transcripts ###
        trainsets = sorted(trainsets, key=lambda x: len(x[2]))
        
        for mode, tmp in zip(['Train', 'Valid'],[trainsets, valsets]):
            # mode = 'debug'+mode
            f1 = open(os.path.join('./dataset/'+mode+".txt"), "w")
            f2 = open(os.path.join('./dataset/'+mode+"_js.txt"), "w")
            
            videos_paths,audios_paths, transcripts = zip(*tmp)
            for video_path, audio_path,transcript in zip(videos_paths, audios_paths, transcripts):
                char_id_transcript = sentence_to_target(transcript, char2id)
                f1.write(f'{video_path}\t{audio_path}\t{transcript}\t{char_id_transcript}\n')
                char_id_transcript = sentence_to_target(convert_to_JASO(transcript), char2id_jaso)
                f2.write(f'{video_path}\t{audio_path}\t{transcript}\t{char_id_transcript}\n')
    else:
        with open(os.path.join('./dataset/'+mode+'.txt'),'w') as f:
            tmp = list(zip(videos_paths,audios_paths,transcripts))
            for a,b,c in tmp:
                f.write(f'{a}\t{b}\t{c}\t{str(0)}\n')



def preprocess(args, test=False):
    data_folder = args.data_folder
    video_path  = args.video_folder if args.video_folder else 'Video_npy'
    
    print('preprocess started..')
    mode = 'Test' if test else 'Train'
    transcripts=[]
    
    dataset_path_video = data_folder + f'/{video_path}/*'
    videos_paths = glob.glob(dataset_path_video)
    videos_paths = sorted(videos_paths)

    dataset_path_audio = data_folder + '/Audio/*.wav'
    audios_paths = glob.glob(dataset_path_audio)
    audios_paths = sorted(audios_paths)
    
    
    if mode=='Train':
        for file_ in tqdm(audios_paths):
            txt_file_ = file_.replace('.wav','.txt')
            txt_file_ = txt_file_.replace('Audio','Text')
            with open(txt_file_, "r",encoding='utf-8') as f:
                raw_sentence = f.read()
            transcripts.append(raw_sentence)
    else:
        transcripts.extend(['-']*len(audios_paths))

    return videos_paths, audios_paths, transcripts



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-v', '--video_folder', type=str)
    parser.add_argument('-s', '--valid_rate', type=float)
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    
    args = get_args()
    test = args.test
    videos_paths, audios_paths, transcripts = preprocess(args, test=test)
    generate_character_script(videos_paths,audios_paths, transcripts, test=test, valid_rate=args.valid_rate)