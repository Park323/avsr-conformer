#-*-coding:utf-8-*-

import re
import csv
import pdb


class Vocabulary(object):
    """
    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, *args, **kwargs):
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.unk_id = None

    def label_to_string(self, labels):
        raise NotImplementedError


class KsponSpeechVocabulary(Vocabulary):
    def __init__(self, vocab_path):
        super(KsponSpeechVocabulary, self).__init__()
        
        self.vocab_dict, self.id_dict = self.load_vocab(vocab_path, encoding='utf-8')
        self.sos_id = int(self.vocab_dict['<sos>'])
        self.eos_id = int(self.vocab_dict['<eos>'])
        self.pad_id = int(self.vocab_dict['<pad>'])
        self.unk_id = int(self.vocab_dict['<unk>'])
        self.labels = self.vocab_dict.keys()

        self.vocab_path = vocab_path
      

    def __len__(self):
        return len(self.vocab_dict)

    def label_to_string(self, labels):
        """
        Converts label to string (number => Hangeul)
        Args:
            labels (numpy.ndarray): number label
        Returns: sentence
            - **sentence** (str or list): symbol of labels
        """

        if len(labels.shape) == 1:
            sentence = str()
            for label in labels:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.unk_id:
                  continue
                sentence += self.id_dict[label.item()]
            return sentence

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.unk_id:
                  continue
                sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def load_vocab(self, label_path, encoding='utf-8'):
        """
        Provides char2id, id2char
        Args:
            label_path (str): csv file with character labels
            encoding (str): encoding method
        Returns: unit2id, id2unit
            - **unit2id** (dict): unit2id[unit] = id
            - **id2unit** (dict): id2unit[id] = unit
        """
        unit2id = dict()
        id2unit = dict()
        
        try:
            with open(label_path, 'r', encoding=encoding) as f:
                labels = csv.reader(f, delimiter=',')
                next(labels)
                
                for row in labels:
                    unit2id[row[1]] = row[0]
                    id2unit[int(row[0])] = row[1]
                
                #unit2id['<blank>'] = len(unit2id)
                #id2unit[len(unit2id)] = '<blank>'

            return unit2id, id2unit
        except IOError:
            raise IOError("Character label file (csv format) doesn`t exist : {0}".format(label_path))


# 유니코드 한글 시작 : 44032, 끝 : 55199
BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def convert(test_keyword):
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
                result.append('# ')
            else:
                result.append(f'#{JONGSUNG_LIST[char3]}')
            #print('종성 : {}'.format(JONGSUNG_LIST[char3]))
        else:
            result.append(keyword)
    # result
    print("".join(result))