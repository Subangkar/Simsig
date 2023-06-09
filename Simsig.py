"""###Model Loading & Patient Database Generation"""

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import dataloaders.deepbeat
# SimCLR
from models.resnext1d import ResNext1D
from models.simclr.simclrmulti_resnext import SimCLRMulti_Resnext

DATA_PATH = 'data'
BATCH_SIZE = 512
MMAP = True
MODEL_FILENAME = 'simclr_ntxentmulti'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

if not os.path.exists('artifacts'):
    os.makedirs('artifacts', exist_ok=True)


class FeatNet(nn.Module):
    def __init__(self, model, skip_last_n=2):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-skip_last_n])

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)


def get_embeddings(model: nn.Module, generator: DataLoader, desc='') -> (np.ndarray, np.ndarray):
    t = tqdm(generator, desc=f"{desc}", total=len(generator))
    embeddings = []
    labels = []
    ids = []
    for step, batch in enumerate(t):
        x, label, id = batch
        x = x.to(device=device, non_blocking=True)
        embedding = model(x)
        embeddings.append(embedding.cpu().numpy())
        labels.append(label.argmax(dim=1))
        ids.append(id)
        # print(embedding.size(), ids, labels)
        # break

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, torch.cat(labels, dim=0).cpu().numpy(), torch.cat(ids, dim=0).cpu().numpy()


# SimCLR model
# model_embd = SimCLR(ResNext1D(), 128, n_features=1024)
model_encoder = SimCLRMulti_Resnext(projection_dim=128)
model_encoder = model_encoder.to(device)
model_encoder.load_state_dict(
    torch.load(f'saved_model/{MODEL_FILENAME}.pt', map_location=device)['state_dict']
)
model_encoder.eval()

model = FeatNet(model=model_encoder, skip_last_n=1).to(device)
model.eval()
''

# Generated Patient Database is saved in `artifacts` folder
with torch.no_grad():
    train_generator = dataloaders.deepbeat.getGenerator(os.path.join(DATA_PATH, 'train'),
                                                        batch_size=BATCH_SIZE, shuffle=False,
                                                        mmap=MMAP)
    X_embds_train, y_train, y_ids = get_embeddings(model, generator=train_generator, desc='train')
    print(X_embds_train.shape)
    np.save('artifacts/X_embds1024_train.npy', X_embds_train)
    np.save('artifacts/y_train.npy', y_train)
    np.save('artifacts/y_train_id.npy', y_ids)

"""###Neighbor Selection & Test Set Result"""

# Test Set Embeddings Generation
with torch.no_grad():
    test_generator = dataloaders.deepbeat.getGenerator(os.path.join(DATA_PATH, 'test'),
                                                       batch_size=BATCH_SIZE, shuffle=False,
                                                       mmap=MMAP)
    X_embds_test, y_test, y_ids = get_embeddings(model, generator=test_generator, desc='test')
    print(X_embds_test.shape)
    np.save('artifacts/X_embds1024_test.npy', X_embds_test)
    np.save('artifacts/y_test.npy', y_test)
    np.save('artifacts/y_test_id.npy', y_ids)

import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm
import gc
import sklearn.metrics
from neighbor_metrics import *

# MODEL_FILENAME = 'simclr_ntxentmulti'
# DATA_PATH = 'data'
EMBEDDING_DIM = 1024

TEST_SET = 'test'
SUPPORT_SET = 'train'

X_test_embd = np.load(f'artifacts/X_embds{EMBEDDING_DIM}_{TEST_SET}.npy')
y_lb_test = np.load(f'artifacts/y_{TEST_SET}.npy')
y_id_test = np.load(f'artifacts/y_{TEST_SET}_id.npy').astype(int)
y_id_test_uniq = np.unique(y_id_test)

X_supp_embd = np.load(f'artifacts/X_embds{EMBEDDING_DIM}_{SUPPORT_SET}.npy')
y_lb = np.load(f'artifacts/y_{SUPPORT_SET}.npy')
y_id = np.load(f'artifacts/y_{SUPPORT_SET}_id.npy').astype(int)
y_id_uniq = np.unique(y_id)

id_to_label_map = ind_to_label_map_generator(y_id, y_lb)
id_to_label_map.update(ind_to_label_map_generator(y_id_test, y_lb_test))

np.set_printoptions(precision=2)


def neighbors_unweighted(k_s, nbr_k_fn):
    count_corr = [0] * (1 + max(k_s))
    count_total = [0] * (1 + max(k_s))
    y_true = [[] for i in range(1 + max(k_s))]
    y_pred = [[] for i in range(1 + max(k_s))]

    t_ind = tqdm(y_id_test_uniq)
    for ind in t_ind:
        X_ind = X_test_embd[y_id_test == ind]
        X_nbr = X_supp_embd
        y_nbr_ids = y_id

        t_ind.set_postfix({"ind_id": ind, "samples": len(X_ind), "neighbors": len(X_nbr)}, refresh=True)

        d_ind = pairwise_distances(X_ind, X_nbr, metric='cosine')

        t_ind.set_postfix({"ind_id": ind, "samples": len(X_ind), "neighbors": len(X_nbr), "d_shape": d_ind.shape},
                          refresh=True)

        k = max(k_s)
        y_nearest_k = nbr_k_fn(d_ind, y_nbr_ids, k)
        y_nearest_labels = ids_to_labels(id_to_label_map, y_nearest_k)

        t_ind.set_postfix({"ind_id": ind, "samples": len(X_ind), "neighbors": len(X_nbr)}, refresh=True)
        for k in k_s:
            count_total[k] += 1
            count_corr[k] += int(id_to_label_map[ind] == 1 and y_nearest_labels[:k].sum() >= (k / 2)) + int(
                id_to_label_map[ind] == 0 and y_nearest_labels[:k].sum() < (k / 2))
            y_true[k].append(int(id_to_label_map[ind] == 1))
            y_pred[k].append(int(y_nearest_labels[:k].sum() >= (k / 2)))

        gc.collect()

    for k in k_s:
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true[k], y_pred[k]).ravel()
        print(f'k={k}',
              f'correct:{count_corr[k]}/{count_total[k]}',
              f'accuracy: {(tn + tp) * 1.0 / (tn + fp + fn + tp):.3f}',
              f'tp:{tp}', f'tn:{tn}', f'fp:{fp}', f'fn:{fn}',
              f'f1: {sklearn.metrics.f1_score(y_true[k], y_pred[k]):.3f}',
              sep=', ')


print(MODEL_FILENAME, 'Average Min:')
neighbors_unweighted(k_s=[7], nbr_k_fn=average_min_k_indiv_ids)


def neighbors_weighted(k_s, nbr_k_fn):
    count_corr = [0] * (1 + max(k_s))
    count_total = [0] * (1 + max(k_s))
    y_true = [[] for i in range(1 + max(k_s))]
    y_pred = [[] for i in range(1 + max(k_s))]

    t_ind = tqdm(y_id_test_uniq)
    for ind in t_ind:
        X_ind = X_test_embd[y_id_test == ind]
        X_nbr = X_supp_embd
        y_nbr_ids = y_id

        t_ind.set_postfix({"ind_id": ind, "samples": len(X_ind), "neighbors": len(X_nbr)}, refresh=True)

        d_ind = pairwise_distances(X_ind, X_nbr, metric='cosine')
        t_ind.set_postfix({"ind_id": ind, "samples": len(X_ind), "neighbors": len(X_nbr), "d_shape": d_ind.shape},
                          refresh=True)

        k = max(k_s)
        y_nearest_k, y_nearest_k_wt = nbr_k_fn(d_ind, y_nbr_ids, k)
        y_nearest_labels = ids_to_labels(id_to_label_map, y_nearest_k)

        wt_pred = np.copy(y_nearest_labels)
        wt_pred[y_nearest_labels == 0] = -1
        wt_pred = (y_nearest_k_wt * wt_pred)

        t_ind.set_postfix({"ind_id": ind, "samples": len(X_ind), "neighbors": len(X_nbr)}, refresh=True)
        for k in k_s:
            is_pred_af = not bool(wt_pred[:k].sum() < 0)
            count_total[k] += 1
            count_corr[k] += int(id_to_label_map[ind] == 1 and is_pred_af) + int(
                id_to_label_map[ind] == 0 and not is_pred_af)
            y_true[k].append(int(id_to_label_map[ind] == 1))
            y_pred[k].append(int(is_pred_af))

        gc.collect()

    for k in k_s:
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true[k], y_pred[k]).ravel()
        print(f'k={k}',
              f'correct:{count_corr[k]}/{count_total[k]}',
              f'accuracy: {(tn + tp) * 1.0 / (tn + fp + fn + tp):.3f}',
              f'tp:{tp}', f'tn:{tn}', f'fp:{fp}', f'fn:{fn}',
              f'f1: {sklearn.metrics.f1_score(y_true[k], y_pred[k]):.3f}',
              sep=', ')


print(MODEL_FILENAME, "Weighted Average Min")
neighbors_weighted(k_s=[7], nbr_k_fn=average_weighted_l2_min_k_indiv_ids)
