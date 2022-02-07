import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('-v','--valid_rate',
                        type=float, help='VALIDATION_RATE')
    parser.add_argument('-d','--data',
                        type=str, help='TRAIN_DATA_PATH')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    
    with open(args.data, 'r') as f:
        data = f.readlines()
    data = np.array(data)
    val_length = int(len(data) * args.valid_rate)
    val_indices = np.sort(np.random.choice(len(data), val_length, replace=False))
    mask  = np.zeros(len(data), dtype=bool)
    mask[val_indices] = True
    with open('dataset/Train.txt', 'w') as f:
        for line in data[~mask]:
            f.write(line)
    with open('dataset/Valid.txt', 'w') as f:
        for line in data[mask]:
            f.write(line)