import os
import cv2
import glob
import numpy as np
import argparse
import pdb
import tqdm


def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    video[1:,:,:,:] -= video[:-1,:,:,:]
    return video[...,::-1]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    data_folder = args.data_folder
    if os.path.exists('./'+data_folder.replace('Video','Video_npy')):
        pass_data = os.listdir('./'+data_folder.replace('Video','Video_npy'))
    else:
        pass_data = []

    filenames = glob.glob(f'{data_folder}/*.mp4')
    filenames = sorted(filenames)
    
    for filename in tqdm.tqdm(filenames):
        if filename.split('\\')[-1][:-4]+'.npz' in pass_data:
            continue
        data = extract_opencv(filename) 
        path_to_save = os.path.join(data_folder.replace('Video','Video_npy'),
                                    filename.split('\\')[-1][:-4])
        if not os.path.exists(os.path.dirname(path_to_save)):
            print(f"{path_to_save} doesn't exist@")
            try:
                os.makedirs(os.path.dirname(path_to_save))
                print(f"Make directory : {path_to_save}")
            except OSError as exc:
                raise
        # np.save(path_to_save, data) # Too Large to store
        np.savez_compressed(path_to_save, video=data) # Compress file to 20%