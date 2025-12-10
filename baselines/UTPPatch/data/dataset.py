import pickle

import numpy as np
from torch.utils.data import Dataset


class BLASTDataset(Dataset):

    def __init__(self, mode, **kwargs) -> None:
        super().__init__()
        self.mode = mode
        if self.mode == 'val':
            self.mode = 'valid'
        assert self.mode in ['train', 'valid', 'test']

        self.pack_length = int(kwargs['pack_length'])
        self.num_valid_samples = kwargs.get('num_valid_samples', None)

        self.li_min = int(kwargs.get('li_min', 48))
        self.li_max = int(kwargs.get('li_max', 1024))

        self.alpha_min = float(kwargs.get('alpha_min', 0.1))
        self.alpha_max = float(kwargs.get('alpha_max', 1.0))

        shape = np.load(f"datasets/BLAST/{self.mode}/shape.npy")
        self.memmap_data = np.memmap(
            f'datasets/BLAST/{self.mode}/data.dat', dtype=np.float32, shape=tuple(shape), mode='r'
        )

        if self.mode == 'valid' and self.num_valid_samples is not None:
            x = self.num_valid_samples
            y = self.memmap_data.shape[0]
            _p = (y - 1) / max(1, (x - 1))
            idx_list = [int(_p * i) for i in range(x)]
            self.memmap_data = self.memmap_data[idx_list]

    def _sample_lengths(self) -> tuple[int, int]:
        li = int(np.random.uniform(self.li_min, self.li_max))
        alpha = np.random.uniform(self.alpha_min, self.alpha_max)
        lo = int(li * alpha)
        return li, lo

    def _get_valid_seq(self) -> np.ndarray:
        while True:
            idx = np.random.randint(0, self.memmap_data.shape[0])
            seq = self.memmap_data[idx].astype(np.float32)
            valid_end = seq.shape[0] - np.argmax(np.flipud(~np.isnan(seq)))
            valid_points = (~np.isnan(seq)).sum()
            if valid_end < 512:
                continue
            if valid_points / max(1, valid_end) < 0.5:
                continue
            return seq

    def _mask_abnormal(self, inputs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.nanmean(inputs)
        std = np.nanstd(inputs)
        if not np.isfinite(std) or std < 1e-3:
            labels = np.ones_like(labels, dtype=np.float32) * mean
            return inputs, labels
        inputs = (inputs - mean) / std
        labels = (labels - mean) / std
        inputs_mask = np.abs(inputs) > 4
        labels_mask = np.abs(labels) > 4
        inputs = inputs.copy()
        labels = labels.copy()
        inputs[inputs_mask] = np.nan
        labels[labels_mask] = np.nan
        return inputs, labels

    def __getitem__(self, idx: int) -> dict:
        seq = self._get_valid_seq()

        x_parts = []
        x_mask_parts = []
        block_split_parts = []
        split_mask_parts = []
        query_mask_parts = []
        labels_parts = []
        labels_mask_parts = []

        total_len = 0

        while total_len < self.pack_length:
            Li, Lo = self._sample_lengths()
            block_len = Li + 1 + Lo

            valid_end = seq.shape[0] - np.argmax(np.flipud(~np.isnan(seq)))
            if valid_end < block_len:
                seq = self._get_valid_seq()
                continue

            start = np.random.randint(0, valid_end - block_len + 1)
            ctx = seq[start:start + Li]
            fut = seq[start + Li:start + Li + Lo]

            ctx, fut = self._mask_abnormal(ctx, fut)

            ctx_mask = (~np.isnan(ctx)).astype(np.float32)
            fut_mask = (~np.isnan(fut)).astype(np.float32)

            ctx_norm = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            fut_norm = np.nan_to_num(fut, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            x_block = np.concatenate([ctx_norm, np.zeros(1, dtype=np.float32), np.zeros(Lo, dtype=np.float32)])
            x_mask_block = np.concatenate([ctx_mask, np.zeros(1, dtype=np.float32), np.zeros(Lo, dtype=np.float32)])

            block_split = np.zeros(block_len, dtype=np.float32)
            block_split[0] = 1.0

            split_mask = np.zeros(block_len, dtype=np.float32)
            split_mask[Li] = 1.0

            query_mask = np.zeros(block_len, dtype=np.float32)
            if Lo > 0:
                query_mask[Li + 1:Li + 1 + Lo] = 1.0

            x_parts.append(x_block)
            x_mask_parts.append(x_mask_block)
            block_split_parts.append(block_split)
            split_mask_parts.append(split_mask)
            query_mask_parts.append(query_mask)
            labels_parts.append(fut_norm)
            labels_mask_parts.append(fut_mask.astype(np.float32))

            total_len += block_len

        x = np.concatenate(x_parts)
        x_mask = np.concatenate(x_mask_parts)
        block_split_mask = np.concatenate(block_split_parts)
        input_output_split_mask = np.concatenate(split_mask_parts)
        query_mask = np.concatenate(query_mask_parts)

        if x.shape[0] > self.pack_length:
            cut = self.pack_length
            x = x[:cut]
            x_mask = x_mask[:cut]
            block_split_mask = block_split_mask[:cut]
            input_output_split_mask = input_output_split_mask[:cut]
            query_mask = query_mask[:cut]

        labels = []
        labels_mask = []
        acc = 0
        for xb, qm, lb, lm in zip(x_parts, query_mask_parts, labels_parts, labels_mask_parts):
            bl = xb.shape[0]
            remaining = max(0, self.pack_length - acc)
            take = min(bl, remaining)
            if take > 0:
                q_count = int(qm[:take].sum())
                if q_count > 0:
                    labels.append(lb[:q_count])
                    labels_mask.append(lm[:q_count])
            acc += bl

        labels = np.concatenate(labels) if len(labels) > 0 else np.zeros(0, dtype=np.float32)
        labels_mask = np.concatenate(labels_mask) if len(labels_mask) > 0 else np.zeros(0, dtype=np.float32)

        # Pad labels and labels_mask to pack_length
        if labels.shape[0] < self.pack_length:
            pad_len = self.pack_length - labels.shape[0]
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=0.0)
            labels_mask = np.pad(labels_mask, (0, pad_len), 'constant', constant_values=0.0)

        return {
            'x': x.astype(np.float32),
            'x_mask': x_mask.astype(np.float32),
            'block_split_mask': block_split_mask.astype(np.float32),
            'input_output_split_mask': input_output_split_mask.astype(np.float32),
            'query_mask': query_mask.astype(np.float32),
            'labels': labels.astype(np.float32),
            'labels_mask': labels_mask.astype(np.float32),
        }

    def __len__(self):
        return self.memmap_data.shape[0]


def demo_visualize():
    import matplotlib.pyplot as plt

    ds = BLASTDataset(mode='train', pack_length=2048,
                      li_min=48, li_max=1024, alpha_min=0.1, alpha_max=1.2,
                      num_valid_samples=1000)
    item = ds[0]
    x = item['x']
    xm = item['x_mask']
    bm = item['block_split_mask']
    sm = item['input_output_split_mask']
    qm = item['query_mask']
    labels = item['labels']
    labels_mask = item['labels_mask']

    fig, ax = plt.subplots(6, 1, figsize=(12, 7), sharex=True)

    def plot_row(arr, label, color='tab:blue'):
        ax_i = ax[0] if label == 'x_mask' else \
               ax[1] if label == 'block_split_mask' else \
               ax[2] if label == 'split_slot_mask' else \
               ax[3] if label == 'query_mask' else \
               ax[4]
        ax_i.fill_between(range(len(arr)), arr, alpha=0.7, color=color)
        ax_i.set_ylabel(label, rotation=0, ha='right', va='center')
        ax_i.set_ylim(-0.1, 1.1)
        ax_i.set_yticks([0, 1])

    plot_row(xm, 'x_mask', 'tab:blue')
    plot_row(bm, 'block_split_mask', 'tab:orange')
    plot_row(sm, 'split_slot_mask', 'tab:green')
    plot_row(qm, 'query_mask', 'tab:red')

    ax_vals = ax[4]
    ax_vals.plot(x, color='tab:purple', marker='o', markersize=1)
    ax_vals.set_ylabel('values (norm)', rotation=0, ha='right', va='center')

    ax_lbl = ax[5]
    query_idx = np.where(qm == 1)[0]
    obs_idx = np.where(labels_mask == 1)[0]
    aligned_idx = query_idx[obs_idx]
    ax_lbl.scatter(aligned_idx, labels[obs_idx],
                   color='tab:red', marker='x', label='labels', s=1)
    ax_lbl.set_ylabel('labels', rotation=0, ha='right', va='center')
    ax_lbl.set_xlabel('position')
    ax_lbl.set_title('Full sequence')

    plt.tight_layout()
    plt.show()
    plt.savefig('demo_visualize.png')


if __name__ == '__main__':
    demo_visualize()
