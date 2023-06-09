# SimSig
Source code & Pretrained models of SimSig, Contrastive Self-Supervised Learning Based Approach for Patient Similarity: A Case Study on Atrial Fibrillation Detection from PPG Signal  
Two Pretrained pytorch model files for CPU is provided in [`saved_model`](saved_model) folder. However, it will utilize GPU if available and the environment is set up. The [`simclr_ntxentmulti.pt`](saved_model/simclr_ntxentmulti.pt) is trained with NT-Xent Multi loss while the [`simclr_ntxent.pt`](saved_model/simclr_ntxent.pt) is trained with NT-Xent loss.

## Requirements
```
Python version 3.7+
PyTorch version 1.8+
```

## How to run:
   - First, set up a virtual environment and activate it
   - Install all the [requirements](requirements.txt) and their dependencies
   - Then download the dataset, redistribute it and put that into proper file & folder structure
   - Finally, run `python Simsig.py` (You can alternatively execute the notebook [`Simsig`](Simsig.ipynb))

**Data Folder Structure for running [`Simsig.py`](Simsig.py):**
```
data/
    train/
        signal.npy
        rhythm.npy
        ids.npy
    test/
        signal.npy
        rhythm.npy
        ids.npy
```
Here, `ids.npy` is derived from corresponding `parameters.npy` for the set that contains individual id for the corresponding segment. The train set is required for generating the **Patient Database** .

## Additional Files:
[distr_split_ids.npy](distr_split_ids.npy): A dictionary that contains list of individal ids for train, validation & test set for the redistribution of dataset according to [BayesBeat: Reliable Atrial Fibrillation Detection from Noisy Photoplethysmography Data](https://dl.acm.org/doi/abs/10.1145/3517247)
