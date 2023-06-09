import os.path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AFdataset(Dataset):
    def __init__(self, ds_npz):
        super().__init__()
        self.ds_npz = ds_npz

    def __getitem__(self, index: int):
        X = self.ds_npz['signal'][index]
        rhythm = self.ds_npz['rhythm'][index]
        id = self.ds_npz['id'][index]
        X, rhythm = np.array(X, dtype='float32').reshape((1, 800)), rhythm.astype('float32')
        return X, rhythm, id

    def __len__(self):
        return len(self.ds_npz['signal'])


def getGenerator(file_location, batch_size, shuffle=True, mmap=True):
    dataset = AFdataset({
        'signal': np.load(os.path.join(file_location, 'signal.npy'), mmap_mode='r' if mmap else None),
        'id': np.load(os.path.join(file_location, 'ids.npy')),
        'rhythm': np.load(os.path.join(file_location, 'rhythm.npy'))
    })
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

