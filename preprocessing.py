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




def generate_character_script(videos_paths, audios_paths, transcripts,mode):
    print('create_script started..')
    if mode == 'Train':
        char2id, id2char = load_label("./dataset/labels.csv")
    
        with open(os.path.join('./dataset/'+mode+".txt"), "w") as f:
            tmp = list(zip(videos_paths, audios_paths, transcripts))
            random.shuffle(tmp)
            videos_paths,audios_paths, transcripts = zip(*tmp)
            for video_path, audio_path,transcript in zip(videos_paths,audios_paths, transcripts):
                char_id_transcript = sentence_to_target(transcript, char2id)
                f.write(f'{video_path}\t{audio_path}\t{transcript}\t{char_id_transcript}\n')
    else:
        with open(os.path.join('./dataset/'+mode+'.txt'),'w') as f:
            tmp = list(zip(videos_paths,audios_paths,transcripts))
            for a,b,c in tmp:
                f.write(f'{a}\t{b}\t{c}\t{str(0)}\n')


    
def preprocess(dataset_path,mode):
    print('preprocess started..')
    transcripts=[]
    dataset_path_video = data_folder + '/Video/*'
    videos_paths = glob.glob(dataset_path_video)
    videos_paths = sorted(videos_paths)

    dataset_path_audio = data_folder + '/Audio/*.wav'
    audios_paths = glob.glob(dataset_path_audio)
    audios_paths = sorted(audios_paths)
    if mode == 'Train':
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
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_args()
    data_folder = args.data_folder
    mode = args.mode
    videos_paths, audios_paths, transcripts = preprocess(data_folder,mode)
    generate_character_script(videos_paths,audios_paths, transcripts, mode)