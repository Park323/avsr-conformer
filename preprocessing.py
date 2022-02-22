import os
import re
import random
from tqdm import tqdm
import glob
import pandas as pd
import pdb
import argparse


def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    freq_list = ch_labels["freq"]

    for (id_, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()
    
    for ch in sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        except KeyError:
            continue

    return target[:-1]



def target_to_sentence(target, id2char):
    sentence = ""
    targets = target.split()

    for n in targets:
        sentence += id2char[int(n)]
    return sentence



def generate_character_script(videos_paths, audios_paths, transcripts, test=False, valid_rate=0.1):
    print('create_script started..')
    mode = 'Test' if test else 'Train'
    if mode == 'Train':
        char2id, id2char = load_label("./dataset/labels.csv")
        
        tmp = list(zip(videos_paths, audios_paths, transcripts))
        
        ### Train/Valid Split ###
        random.shuffle(tmp)
        val_num = int(valid_rate*len(tmp))
        trainsets = tmp[:-val_num]
        valsets = tmp[-val_num:]
    
        ### Sort Train Set ###
        def get_transcripts_length(dataset):
            return len(dataset[2])
        trainsets = sorted(tmp, key=get_transcripts_length)
        
        for mode, tmp in zip(['Train', 'Valid'],[trainsets, valsets]):
            with open(os.path.join('./dataset/'+mode+".txt"), "w") as f:
                videos_paths,audios_paths, transcripts = zip(*tmp)
                for video_path, audio_path,transcript in zip(videos_paths,audios_paths, transcripts):
                    char_id_transcript = sentence_to_target(transcript, char2id)
                    f.write(f'{video_path}\t{audio_path}\t{transcript}\t{char_id_transcript}\n')
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