import os
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler
from tqdm.auto import tqdm


class AFPairedDataset(Dataset):
    def __init__(self, signal, ind_ids):
        super().__init__()
        self.signal = signal
        self.ind_ids = ind_ids

    def __getitem__(self, index: list):
        index = (index[0] // 2) % len(self.ind_ids)
        index = np.random.choice(self.ind_ids[index][1], 2, replace=False)
        X = self.signal[index]
        X = np.array(X, dtype='float32').reshape((-1, 1, 800))
        x_i = X[0]
        x_j = X[1]
        return x_i, x_j

    def __len__(self):
        return len(self.signal)


def getSimPairLoader(file_location, batch_size, mmap=True):
    signal = np.load(os.path.join(file_location, 'signal.npy'), mmap_mode='r' if mmap else None)
    ids = np.load(os.path.join(file_location, 'ids.npy'))
    ind_ids = np.load(os.path.join(file_location, 'ind_ids.npy'), allow_pickle=True)

    return DataLoader(dataset=AFPairedDataset(signal, ind_ids),
                      sampler=BatchSampler(SequentialSampler(range(len(ids))),
                                           batch_size=2,
                                           drop_last=False),
                      batch_size=batch_size,
                      num_workers=0,
                      pin_memory=True)


if __name__ == '__main__':
    # l = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = 23
    # file_location = 'data/test'
    # signal = np.load(os.path.join(file_location, 'signal.npy'), mmap_mode='r')
    # print(len(signal))
    # # ids = np.load(os.path.join(file_location, 'ids.npy'))
    # # indices = np.concatenate([idx_b for idx_b in tqdm(get_pos_neg_indices_generator(signal, ids, b),
    # #                                                   total=int(math.ceil(signal.shape[0] / (2 * b))),
    # #                                                   desc=f'Generating {file_location} Batch Indices')])
    # # print(len(signal), len(indices))
    # # dl = DataLoader(dataset=AFPairedDataset(l),
    # #                 sampler=BatchSampler(indices,
    # #                                      batch_size=2,
    # #                                      drop_last=False),
    # #                 batch_size=b,
    # #                 num_workers=0,
    # #                 pin_memory=True)
    # # print(len(dl))
    # dl = getSimPairLoader(file_location, b)
    st = time.time()
    # for i, batch in enumerate(tqdm(dl)):
    #     print(i, len(batch[0]), batch)
    #     # break
    #     # x_i, x_j = batch
    #     # print(x_i.size(), x_j.size())
    #     # print(i, batch)
    #     pass
    # print(time.time() - st, len(dl))
    pass
