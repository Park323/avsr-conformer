#-*-coding:utf-8-*-

import os
import re
import random
from tqdm import tqdm
import glob
import pandas as pd
import pdb
import argparse

# 유니코드 한글 시작 : 44032, 끝 : 55203
BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28
END_CODE = 55203

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = ['#', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def convert_to_JASO(test_keyword):
    split_keyword_list = list(test_keyword)
    #print(split_keyword_list)

    result = list()
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - BASE_CODE
            char1 = int(char_code / CHOSUNG)
            result.append(CHOSUNG_LIST[char1])
            #print('초성 : {}'.format(CHOSUNG_LIST[char1]))
            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            result.append(JUNGSUNG_LIST[char2])
            #print('중성 : {}'.format(JUNGSUNG_LIST[char2]))
            char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
            if char3==0:
                result.append('##')
            else:
                result.append(f'#{JONGSUNG_LIST[char3]}')
            #print('종성 : {}'.format(JONGSUNG_LIST[char3]))
        else:
            result.append(keyword)
    # result
    return result


def convert_to_char(JASOlist):
    id2KR = {code-BASE_CODE+5:chr(code)
             for code in range(BASE_CODE ,END_CODE + 1)}
    id2KR[0] = '<pad>'
    id2KR[1] = '<sos>'
    id2KR[2] = '<eos>' 
    id2KR[3] = '<unk>'
    id2KR[4] = ' '
    KR2id = {key:value for value, key in id2KR.items()}
    
    result = list()
    chr_count = 0
    chr_id = 5

    lists = [CHOSUNG_LIST, JUNGSUNG_LIST, JONGSUNG_LIST]
    nums = [CHOSUNG, JUNGSUNG, 1]
    
       
    for JS in JASOlist:
        JS = JS[1:] if JS[0]=='#' else JS
        if JS in ['<pad>', '<sos>', '<eos>', '<unk>', ' ']:
            chr_id = KR2id[JS]
        else:
            try:
                chr_id += lists[chr_count].index(JS) * nums[chr_count]
                chr_count += 1
            except:
                chr_count = 3
                chr_id = KR2id['<unk>']
        
        if chr_count == 3:
            result.append(chr_id)
            if JS=='<eos>':
                return result
            chr_count = 0
            chr_id = 5
    
    return result

def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    freq_list = ch_labels.get("freq", id_list)

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
        
        dataset = list(zip(videos_paths, audios_paths, transcripts))
        
        ### Train/Valid Split ###
        random.shuffle(dataset)
        val_num = int(valid_rate*len(dataset))
        trainsets = dataset[:-val_num]
        valsets = dataset[-val_num:]
    
        ### Sort Train Set by the length of transcripts ###
        trainsets = sorted(trainsets, key=lambda x: len(x[2]))
        
        for mode, tmp in zip(['Train', 'Valid'],[trainsets, valsets]):
            with open(os.path.join('./dataset/'+mode+".txt"), "w") as f:
                videos_paths,audios_paths, transcripts = zip(*tmp)
                for video_path, audio_path,transcript in zip(videos_paths,audios_paths, transcripts):
                    # char_id_transcript = sentence_to_target(transcript, char2id)
                    char_id_transcript = sentence_to_target(convert_to_JASO(transcript), char2id)
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