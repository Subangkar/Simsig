import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler
from tqdm.auto import tqdm


class AFNeighborDataset(Dataset):
    def __init__(self, signal, ids):
        super().__init__()
        self.signal = signal
        self.ids = ids

    def __getitem__(self, index: int):
        X = self.signal[index]
        X = np.array(X, dtype='float32').reshape((1, 800))
        Id = self.ids[index]
        return X, Id

    def __len__(self):
        return len(self.signal)


def collate_fn_simmulti(data):
    X = torch.Tensor(np.stack(list(map(lambda x: x[0], data))))
    Id = torch.IntTensor(list(map(lambda x: x[1], data)))
    adj_mat = []
    for i in Id:
        adj_mat.append(Id == i)
    adj_mat = torch.stack(adj_mat, dim=0)  # stack for compatibility issues with older versions
    return X, adj_mat.float()


def getSimNeighborLoader(file_location, batch_size, shuffle=False, mmap=True):
    signal = np.load(os.path.join(file_location, 'signal.npy'), mmap_mode='r' if mmap else None)
    ids = np.load(os.path.join(file_location, 'ids.npy'))

    return DataLoader(dataset=AFNeighborDataset(signal, ids), collate_fn=collate_fn_simmulti,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=0,
                      pin_memory=True)


if __name__ == '__main__':
    # l = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # b = 8
    # file_location = 'data/test'
    # signal = np.load(os.path.join(file_location, 'signal.npy'), mmap_mode='r')
    # print(len(signal))
    # # ids = np.load(os.path.join(file_location, 'ids.npy'))
    # # indices = np.concatenate([idx_b for idx_b in tqdm(get_pos_neg_indices_generator(signal, ids, b),
    # #                                                   total=int(math.ceil(signal.shape[0] / (2 * b))),
    # #                                                   desc=f'Generating {file_location} Batch Indices')])
    # # print(len(signal), len(indices))
    # # dl = DataLoader(dataset=AFNeighborDataset(l),
    # #                 sampler=BatchSampler(indices,
    # #                                      batch_size=2,
    # #                                      drop_last=False),
    # #                 batch_size=b,
    # #                 num_workers=0,
    # #                 pin_memory=True)
    # # print(len(dl))
    # dl = getSimNeighborLoader(file_location, b, shuffle=True)
    # st = time.time()
    # for i, batch in enumerate(tqdm(dl)):
    #     print(i, len(batch[0]), batch[1], '\n', batch[2])
    #     break
    #     # print(i, batch)
    #     pass
    # print(time.time() - st, len(dl))
    # getSimNeighborLoader('test', 4)
    pass
