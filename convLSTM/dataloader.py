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

    def __init__(self, paths, labels, seq_len, mask, transform=None, size=(200, 200)):
        self.input_path = paths
        self.labels = labels
        self.seq_len = seq_len
        self.mask = mask
        self.transform = transform # Not Implemented
        self.size = size
        self._make_dict()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_folder_path, label = self._select_class(idx)
        frame_paths = glob.glob(os.path.join(video_folder_path, '*.*'))
        frame_paths.sort(key=lambda x: int(x.split('/')[-1].split('-')[0])) # order of squence images
        images = [imageio.imread(path) for path in frame_paths[:self.seq_len]]
        images = [cv2.resize(img, self.size) for img in images]
        image = [np.zeros_like(img) for i, img in enumerate(images) if np.random.randint(0, self.seq_len) < self.seq_len * self.mask]
        images = [np.transpose(img, (2, 0, 1)) for img in images]
        images = torch.FloatTensor(images).cuda()
        label = torch.FloatTensor([label]).cuda()
        return images, label


    def _make_dict(self):
        self.class_dict = defaultdict(lambda: [])
        self.length_dict = {}
        self.class_idx = [-1] * 2
        for i in range(len(self.labels)):
            if len(glob.glob(os.path.join(self.input_path[i], '*.*'))) < self.seq_len: continue
            self.class_dict[self.labels[i]].append(self.input_path[i])

        for k in self.class_dict.keys():
            self.length_dict[k] = len(self.class_dict[k])
        print('---Done, Class dict!---')

    def _select_class(self, idx):
        if idx % 2 == 0:
            self.class_idx[0] += 1
            return self.class_dict[0][self.class_idx[0] % self.length_dict[0]], 0
        else:
            self.class_idx[1] += 1
            return self.class_dict[1][self.class_idx[1] % self.length_dict[1]], 1
        