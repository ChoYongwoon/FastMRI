import h5py
import random
import torch
import numpy as np
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SliceData(Dataset):
    def __init__(self, roots, transform, input_key, target_key, forward=False, train = False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.examples = []
        self.train = train
        
        files = list(Path(roots[0]).iterdir())
        for fname in sorted(files):
            num_slices = self._get_metadata(roots[0] + fname.name)
            self.examples += [
                (roots, fname.name, slice_ind) for slice_ind in range(num_slices)
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, 'r') as hf:
            num_slices = []
            for key in self.input_key:
                num_slices.append(hf[key].shape[0])
            assert num_slices.count(num_slices[0]) == len(num_slices)
            num_slices = num_slices[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        roots, fname, dataslice = self.examples[i]
        input = []
        with h5py.File(roots[0] + fname, "r") as hf:
            input = [hf[key][dataslice] for key in self.input_key]
            if self.forward:
                target = -1
                attrs = {}
            else:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        with h5py.File(roots[1] + fname, "r") as hf:
            input.append(hf['reconstruction'][dataslice])
            
        input, target, attrs, _, _ = self.transform(np.array(input), target, attrs, fname, dataslice, self.train)

        return input, target, attrs, fname, dataslice

def create_data_loaders(data_path, args, shuffle=False, isforward=False, train=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        roots=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward,
        train = train
    )
    
    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    
    return data_loader
