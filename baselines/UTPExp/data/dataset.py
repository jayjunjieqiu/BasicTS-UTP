import pickle

import numpy as np
from torch.utils.data import Dataset


class BLASTDatasetMixUp(Dataset):

    def __init__(self, mode: str, num_valid_samples: int = None, k_max: int = 3, alpha : float = 1.5, postfix: str = None, **kwargs) -> None:
        super().__init__()
        assert mode in ['train', 'valid', 'test', 'val']
        if mode == 'val': mode = 'valid'
        
        # hyperparameters
        
        # minimum valid history sequence length
        self.min_seq_length = 48
        # minimum valid future sequence length
        self.min_future_length = 16
        
        self.context_length = kwargs['context_length']
        self.target_length = kwargs['target_length']

        self.pad_length = self.context_length - self.min_seq_length
        self.pad_future_length = self.target_length - self.min_future_length

        # parameters for mixup
        self.mode = mode
        self.alpha = alpha
        self.num_valid_samples = num_valid_samples
        self.k_max = k_max

        # load data
        shape = np.load(f"datasets/BLAST/{self.mode}/shape.npy")
        self.memmap_data = np.memmap(f'datasets/BLAST/{self.mode}/data.dat', dtype=np.float32, shape=tuple(shape), mode='r')

        if self.mode == 'valid' and self.num_valid_samples is not None:
            # use only a subset of the validation set to speed up training
            print(f"Using {self.num_valid_samples} samples for {self.mode} dataset")
            x = self.num_valid_samples
            y = self.memmap_data.shape[0]
            _p = (y - 1) / (x - 1)
            idx_list = list(range(self.num_valid_samples))
            idx_list = [int(_p * i) for i in idx_list]
            self.memmap_data = self.memmap_data[idx_list]

        print(f"Loaded {self.mode} dataset with shape {self.memmap_data.shape}")

    def mask_abnormal(self, inputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        # mask abnormal values with nan

        # zscore normalization
        mean = np.nanmean(inputs)
        std = np.nanstd(inputs)
        if std < 1e-3:
            labels = np.ones_like(labels) * mean
            return inputs, labels
        inputs = (inputs - mean) / std
        labels = (labels - mean) / std
        inputs_mask = np.abs(inputs) > 4
        labels_mask = np.abs(labels) > 4
        inputs[inputs_mask] = np.nan
        labels[labels_mask] = np.nan
        return inputs, labels

    def padding_nan(self, seq: np.ndarray) -> np.ndarray:
        # pad the sequence with nan
        seq = np.pad(seq, (self.pad_length, 0), 'constant', constant_values=np.nan)
        seq = np.pad(seq, (0, self.pad_future_length), 'constant', constant_values=np.nan)
        return seq

    def get_valid_end_idx(self, seq: np.ndarray, sample_length: int) -> int:
        if not np.isnan(seq[-1]):
            return seq.shape[0] - sample_length
        else:
            last_non_nan_index = seq.shape[0] - np.argmax(np.flipud(~np.isnan(seq)))
            if last_non_nan_index > sample_length:
                return last_non_nan_index - sample_length
            else:
                raise ValueError("No valid end index found in the sequence")

    def get_valid_seq(self):
        # random select a valid sequence from the memmap data
        while True:
            random_idx = np.random.randint(0, self.memmap_data.shape[0])
            seq = self.memmap_data[random_idx].astype(np.float32)
            valid_length = seq.shape[0] - np.argmax(np.flipud(~np.isnan(seq)))
            valid_point = (~np.isnan(seq)).sum()
            if valid_length < 1000:
                continue
            if self.min_seq_length + self.min_future_length > valid_length:
                continue
            if valid_point / valid_length < 0.5:
                continue
            else:
                return seq, random_idx

    def mixup(self):
        # sampling
        k = np.random.randint(1, self.k_max + 1)
        sampled_indices = np.random.choice(len(self), size=(k), replace=True)
        weights = np.random.dirichlet([self.alpha] * k).astype(np.float32)
        time_series_sampled = self.memmap_data[sampled_indices].astype(np.float32)
        
        # normalize data
        time_series_sampled = np.nan_to_num(time_series_sampled, nan=0., posinf=0., neginf=0.)
        time_series_sampled = (time_series_sampled - np.nanmean(time_series_sampled, axis=1, keepdims=True)) / (np.nanstd(time_series_sampled, axis=1, keepdims=True) + 1e-8)
        
        augmented_batch = np.dot(weights, time_series_sampled)
        return augmented_batch

    def __getitem__(self, idx: int) -> tuple:

        target_seq_len = self.context_length + self.target_length

        seq, random_idx = self.get_valid_seq()

        seq = self.padding_nan(seq)
        random_t = np.random.randint(0, self.get_valid_end_idx(seq, target_seq_len + 1))

        seq = seq[random_t:random_t + target_seq_len]
        
        inputs = seq[:self.context_length]
        labels = seq[self.context_length:self.context_length+self.target_length]
        inputs, labels = self.mask_abnormal(inputs, labels)
        
        return {'inputs': inputs, 'labels': labels}

    def __len__(self):
        return self.memmap_data.shape[0]
