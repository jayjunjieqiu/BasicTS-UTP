import numpy as np
from torch.utils.data import Dataset

class BLASTDataset(Dataset):
    def __init__(self, mode: str, num_valid_samples: int = None, **kwargs) -> None:
        super().__init__()
        assert mode in ['train', 'valid', 'val']
        if mode == 'val': mode = 'valid'
        self.mode = mode
        self.num_valid_samples = num_valid_samples

        # hyperparamters
        self.context_length = kwargs['context_length']
        self.target_length = kwargs['target_length']

        # minimum valid history sequence length
        self.min_context_length = 48
        # minimum valid future sequence length
        self.min_future_length = 16
        # minimum valid sequence length
        self.min_seq_length = self.min_context_length + self.min_future_length
        # sequence length
        self.seq_length = self.context_length + self.target_length
        # padding length for context sequence
        self.pad_context_length = self.context_length - self.min_context_length
        # padding length for future sequence
        self.pad_future_length = self.target_length - self.min_future_length

        # threshold for valid sequence length to avoid ValueError in get_valid_end_idx
        self.min_valid_len_threshold = self.min_context_length + self.target_length + 2

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
    
    def get_valid_seq(self):
        """
        Randomly select a valid sequence from the memmap data.
        """
        while True:
            random_idx = np.random.randint(0, self.memmap_data.shape[0])
            seq = self.memmap_data[random_idx].astype(np.float32)
            # Find the last non-NaN value in the sequence 
            # by flipping the array and locating the first non-NaN index
            last_non_nan_idx = seq.shape[0] - np.argmax(np.flipud(~np.isnan(seq)))
            valid_point = (~np.isnan(seq)).sum()

            if last_non_nan_idx < self.min_valid_len_threshold:
                continue
            elif self.min_seq_length > last_non_nan_idx:
                continue
            elif valid_point / last_non_nan_idx < 0.5:
                continue
            else:
                return seq, random_idx

    def padding_nan(self, seq: np.ndarray) -> np.ndarray:
        """
        Pad the sequence with nan.
        """
        seq = np.pad(seq, (self.pad_context_length, 0), 'constant', constant_values=np.nan)
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
                # This should not happen, as we ensure the sequence length is long enough through `get_valid_seq`
                raise ValueError("No valid end index found in the sequence")

    def mask_abnormal(self, inputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Mask abnormal values in the sequence and normalize the sequence.
        """
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

    def __getitem__(self, idx: int) -> tuple:
        """
        Get the item at the given index.
        """
        seq, _ = self.get_valid_seq()
        seq = self.padding_nan(seq)

        # By choosing the random_t in [0, get_valid_end_idx(seq, self.min_seq_length + 1)],
        # we ensure that the sequence length is at least self.min_seq_length.
        # This guarantees:
        # 1. The valid input sequence length is in [self.min_context_length, self.context_length].
        # 2. The valid output sequence length is in [self.min_future_length, self.target_length].
        random_t = np.random.randint(0, self.get_valid_end_idx(seq, self.seq_length + 1))

        seq = seq[random_t: random_t + self.seq_length]

        # split input and label
        inputs = seq[:self.context_length]
        labels = seq[self.context_length:self.context_length+self.target_length]
        inputs, labels = self.mask_abnormal(inputs, labels)

        # generate mask
        input_mask = np.logical_not(np.isnan(inputs))
        target_mask = np.logical_not(np.isnan(labels))

        # Fill NaNs in inputs and labels with 0 (since mask handles it)
        inputs = np.nan_to_num(inputs, nan=0.0)
        labels = np.nan_to_num(labels, nan=0.0)

        return {'inputs': inputs, 'labels': labels, 'input_mask': input_mask, 'target_mask': target_mask}
    
    def __len__(self):
        return self.memmap_data.shape[0]