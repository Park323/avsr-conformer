import os
import re
import random
from tqdm import tqdm
import glob
import pandas as pd
import numpy as np
import pdb
import argparse



def main(args):
    with open(args.data.replace('.txt', '_.txt'), 'w') as n:
        with open(args.data, 'r') as f:
            for line in f.readlines():
                n.write(line.replace(args.origin, args.target))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-o', '--origin', type=str, default='data/Train')
    parser.add_argument('-t', '--target', type=str, required=True)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    
    args = get_args()
    main(args)