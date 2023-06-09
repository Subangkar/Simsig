import numpy as np
import pandas as pd


def get_unique_N(iterable, N):
    """returns the first N unique elements of iterable.
    Might yield less if data too short."""
    seen = set()
    elems = []
    for e in iterable:
        if e in seen:
            continue
        seen.add(e)
        elems.append(e)
        if len(seen) == N:
            return elems
    return elems


def using_complex(a):
    weight = 1j * np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    b = a + weight[:, np.newaxis]
    u, ind = np.unique(b, return_index=True)
    b = np.zeros_like(a)
    np.put(b, ind, a.flat[ind])
    return b


def reduce2drows_to_uniqe(a, n_unique):
    dup = using_complex(a)
    dup_ = np.zeros((len(dup), n_unique), dtype=int)
    for i, row in enumerate(dup):
        dup_[i] = row[row != 0]
    return dup_


def segwise_min_k_indiv_ids(d: np.array, y_nbr_ids: np.array, k=5):
    sorted_indices = np.argsort(d, axis=1, kind='quicksort')
    d_nearest = d[:, sorted_indices]
    y_nearest = y_nbr_ids[sorted_indices]
    nearest_k_indices = reduce2drows_to_uniqe(y_nearest, len(y_id_uniq) - 1)
    y_nearest_k = nearest_k_indices[:, :k]
    return y_nearest_k


def average_min_k_indiv_ids(d: np.ndarray, y_nbr_ids: np.ndarray, k=5):
    dist_id_pairs = []
    d_flat = d.ravel()
    y_uniq = np.unique(y_nbr_ids)
    y_nbr_ids = np.tile(y_nbr_ids, len(d))
    for ind in y_uniq:
        dist_id_pairs.append((d_flat[y_nbr_ids == ind].mean(), ind))
    y_nearest = np.array([x[1] for x in sorted(dist_id_pairs)])
    return y_nearest[:k]


def average_weighted_l1_min_k_indiv_ids(d: np.ndarray, y_nbr_ids: np.ndarray, k=5):
    dist_id_pairs = []
    d_flat = d.ravel()
    y_uniq = np.unique(y_nbr_ids)
    y_nbr_ids = np.tile(y_nbr_ids, len(d))
    for ind in y_uniq:
        dist_id_pairs.append((d_flat[y_nbr_ids == ind].mean(), ind))
    y_nearest = np.array([x[1] for x in sorted(dist_id_pairs)])
    y_nearest_wights = np.array([1 / x[0] for x in sorted(dist_id_pairs)])
    return y_nearest[:k], y_nearest_wights[:k]


def average_weighted_l2_min_k_indiv_ids(d: np.ndarray, y_nbr_ids: np.ndarray, k=5):
    dist_id_pairs = []
    d_flat = d.ravel()
    y_uniq = np.unique(y_nbr_ids)
    y_nbr_ids = np.tile(y_nbr_ids, len(d))
    for ind in y_uniq:
        dist_id_pairs.append((d_flat[y_nbr_ids == ind].mean(), ind))
    y_nearest = np.array([x[1] for x in sorted(dist_id_pairs)])
    y_nearest_wights = np.array([1 / (x[0] * x[0]) for x in sorted(dist_id_pairs)])
    return y_nearest[:k], y_nearest_wights[:k]


def pct_sort_min_k_indiv_ids(d: np.ndarray, y_nbr_ids: np.ndarray, k=5, radius=0.3):
    dist_id_pairs = []
    d_flat = d.ravel()
    y_uniq = np.unique(y_nbr_ids)
    y_nbr_ids = np.tile(y_nbr_ids, len(d))
    for ind in y_uniq:
        count_satis = (d_flat[y_nbr_ids == ind] < radius).sum()
        count_total = (y_nbr_ids == ind).sum()
        dist_id_pairs.append((
            100 * count_satis / count_total,
            ind))
    y_nearest = np.array([x[1] for x in sorted(dist_id_pairs, reverse=True)])
    return y_nearest[:k]


def pct_sort_count_weighted_min_k_indiv_ids(d: np.ndarray, y_nbr_ids: np.ndarray, k=5, radius=0.3):
    dist_id_pairs = []
    d_flat = d.ravel()
    y_uniq = np.unique(y_nbr_ids)
    y_nbr_ids = np.tile(y_nbr_ids, len(d))
    for ind in y_uniq:
        count_satis = (d_flat[y_nbr_ids == ind] < radius).sum()
        count_total = (y_nbr_ids == ind).sum()
        dist_id_pairs.append((
            100 * count_satis / count_total,
            count_satis,
            ind))
    y_nearest = np.array([x[2] for x in sorted(dist_id_pairs, reverse=True)])
    y_nearest_wights = np.array([x[1] for x in sorted(dist_id_pairs, reverse=True)])
    return y_nearest[:k], y_nearest_wights[:k]


def overall_min_k_indiv_ids(d: np.array, y_nbr_ids: np.array, k=5):
    d_nearest = d.ravel()
    y_nearest = np.tile(y_nbr_ids, len(d))
    sorted_indices = np.argsort(d_nearest, axis=0, kind='quicksort')
    d_nearest = d_nearest[sorted_indices]
    y_nearest = y_nearest[sorted_indices]
    y_nearest = pd.unique(y_nearest)
    return y_nearest[:k]


def ind_to_label_map_generator(y_ids, y_lb) -> dict:
    dct = {}
    for ind in np.unique(y_ids):
        dct[ind] = y_lb[y_ids == ind][0]
    return dct


def ids_to_labels(id_to_lb_map, ids):
    lb = np.zeros_like(ids)
    for i, ind in enumerate(ids):
        lb[i] = id_to_lb_map[ind]
    return lb
