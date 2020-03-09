import torch
import imageio
import cv2
import os
import glob
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np


class Dataloader(Dataset):
    '''
    Keep in mind! the sequences of video must be ordered.

    Not implemented
    1. one frame two people
    '''

    def __init__(self, class_dict, length_dict, class_idx, is_train, cfg, transform=None):
        self.class_dict = class_dict
        self.length_dict = length_dict
        self.class_idx = class_idx
        self.is_train = is_train
        self.cfg = cfg
        self.seq_len = cfg['seq_len']
        self.mask = cfg['mask'] if self.is_train else -1
        self.transform = transform # Not Implemented
        self.size = cfg['input_size']

    def __len__(self):
        return self.cfg['train_iter'] if self.is_train else self.length_dict[0] + self.length_dict[1]

    def __getitem__(self, idx):
        try:
            video_folder_path, label = self._select_class(idx)
            frame_paths = glob.glob(os.path.join(video_folder_path, '*.*'))
            frame_paths.sort(key=lambda x: int(x.split('/')[-1].split('_')[1])) # order of squence images
            frame_paths = [path for path in frame_paths if path.split('/')[-1].split('_')[-1].split('.')[0] == str(0)]
            start = np.random.randint(0, len(frame_paths) - self.seq_len)
            images = [imageio.imread(path) for path in frame_paths[start:start+self.seq_len]]
        except:
            print("Error folder name ", video_folder_path)
            # print('error')
            video_folder_path, label = '/hdd2/dfdc_dataset/3_images1/1_CROPPED/train/REAL/aawvkuxypy', 0
            frame_paths = glob.glob(os.path.join(video_folder_path, '*.*'))
            frame_paths.sort(key=lambda x: int(x.split('/')[-1].split('_')[1])) # order of squence images
            frame_paths = [path for path in frame_paths if path.split('/')[-1].split('_')[-1].split('.')[0] == str(0)]
            start = np.random.randint(0, len(frame_paths) - self.seq_len)
            images = [imageio.imread(path) for path in frame_paths[start:start+self.seq_len]]
            
        
        if self.transform:
            images = [self.transform(image=img) for img in images]
            
        images = [cv2.resize(img, self.size) for img in images]
        image = [np.zeros_like(img) for i, img in enumerate(images) if np.random.randint(0, self.seq_len) < self.seq_len * self.mask]
        images = [np.transpose(img, (2, 0, 1)) for img in images]
        images = torch.FloatTensor(images).cuda()
        label = torch.FloatTensor([label]).cuda()
        return images, label

    def _select_class(self, idx):
        if idx % 2 == 0:
            self.class_idx[0] += 1
            if self.class_idx[0] == self.length_dict[0]:
                self.class_idx[0] = 0
                np.random.shuffle(self.class_dict[0])
            return self.class_dict[0][self.class_idx[0]], 0
        else:
            self.class_idx[1] += 1
            if self.class_idx[1] == self.length_dict[1]:
                self.class_idx[1] = 0
                np.random.shuffle(self.class_dict[1])
            return self.class_dict[1][self.class_idx[1]], 1

