import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import pickle
import re
import glob
import os.path
from PIL import Image, ImageSequence, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


import numpy as np

class TGIF(Dataset):
    def __init__(self, root, preload=False, transform=None, max_frames=30, verbose=False):
        self.root = root
        self.filenames = glob.glob(os.path.join(root, '*.gif'))
        self.transform = transform
        self.max_frames = max_frames
        self.gifs = None
        self.verbose = verbose

        if preload:
            self._preload()
    
    def _preload(self):
        pass
       

    def _file_to_gif_tensor(self, filename):
        gif = Image.open(filename)
        frames = [self.transform(frame.copy().convert('RGB')) for frame in ImageSequence.Iterator(gif)]
        while len(frames) < self.max_frames:
              frames += frames[::-1]

        frames = frames[:self.max_frames]
        return torch.stack(frames)

    def __getitem__(self, index):
        if self.gifs is not None:
            gif = self.gifs[index]
        else:
            gif = self._file_to_gif_tensor(self.filenames[index])

        return gif
            

    def __len__(self):
        return len(self.filenames)
   
class MMNist(Dataset):

    GIF_LENGTH = 20
    GIF_SIZE = 64

    def __init__(self, data_file, transform=None, preload=True, verbose=False, temporality=False, len_lim=None, is_mmnist=True):
        
        self.transform = transform
        self.temporality = temporality
        self.len_lim = len_lim
        self.is_mmnist = is_mmnist

        if preload:
            self.load_data(data_file)

        self.verbose = verbose
        
    def load_data(self, data_file):
        data = np.load(data_file)["arr_0"] if self.is_mmnist else np.load(data_file)

        if self.temporality:
            self._len = data.shape[0] // self.GIF_LENGTH
            self.data = data.reshape(self._len, self.GIF_LENGTH, 1, self.GIF_SIZE, self.GIF_SIZE)
        
            self.data = np.transpose(self.data, (0, 1, 4, 3, 2))
        else:
            if self.len_lim:
                data = data[:self.len_lim]
            self._len = data.shape[0]
            self.data = data.reshape(self._len, 1, self.GIF_SIZE, self.GIF_SIZE)
        
            self.data = np.transpose(self.data, (0, 3, 2, 1))


    def __getitem__(self, index):
        if self.temporality:
            if self.transform:
                gif = self.data[index]
                ret = [self.transform(frame) for frame in self.data[index]]
                return torch.stack(ret)

            ret = self.data[index]
            return ret
        else:
            img = self.data[index]
            if self.transform:
                img = self.transform(img)
            return img


    def __len__(self):
        return self._len

class NumpyGIFs(Dataset):
    GIF_LENGTH = 20
    GIF_SIZE = 64

    def __init__(self, data_file, transform=None, preload=True, verbose=False, temporality=False):
        self.transform = transform
        self.temporality = temporality
        
        if preload:
            self.load_data(data_file)

        self.verbose = verbose
        
    def load_data(self, data_file):
        print (f'Loading {data_file}...')
        data = np.load(data_file)
        print (f'Loaded {len(data)} examples.')
        
        if self.temporality:
            self._len = data.shape[0]
            self.data = data.reshape(self._len, self.GIF_LENGTH, 1, self.GIF_SIZE, self.GIF_SIZE)
            self.data = np.transpose(self.data, (0, 1, 4, 3, 2))
            
        else:
            self._len = data.shape[0]
            self.data = data.reshape(self._len, 1, self.GIF_SIZE, self.GIF_SIZE)
            self.data = np.transpose(self.data, (0, 3, 2, 1))

    def __getitem__(self, index):
        if self.temporality:
            if self.transform:
                return torch.stack([self.transform(frame) for frame in self.data[index]])
            return self.data[index]
        else:
            if self.transform:
                return self.transform(self.data[index])
            return self.data[index]

    def __len__(self):
        return self._len