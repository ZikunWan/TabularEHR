import random
from typing import Iterable, List, Optional

from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler
from transformers import Trainer
from transformers.trainer_utils import seed_worker


def _unwrap_base_dataset(dataset):
    # Unwrap lightweight adapters (e.g., LabelFieldAdapter) to access raw metadata.
    while hasattr(dataset, "base_dataset"):
        dataset = dataset.base_dataset
    return dataset


def infer_sample_lengths(dataset, fallback_from_getitem: bool = True) -> List[int]:
    """
    Infer per-sample sequence lengths for bucketing/sampling.
    Priority:
    1) metadata in sample_info/list_data;
    2) preprocessed cache (`data`) fields;
    3) optional on-the-fly __getitem__ fallback.
    """
    ds = _unwrap_base_dataset(dataset)
    n = len(dataset)
    lengths: List[int] = [0] * n

    # 1) from sample_info style metadata
    if hasattr(ds, "sample_info") and isinstance(ds.sample_info, list) and len(ds.sample_info) == n:
        for i, meta in enumerate(ds.sample_info):
            if not isinstance(meta, dict):
                continue
            if "table_length" in meta:
                lengths[i] = max(1, int(meta["table_length"]))
            elif "context_begin" in meta and "context_end" in meta:
                lengths[i] = max(1, int(meta["context_end"]) - int(meta["context_begin"]))
            elif "period_begin" in meta and "period_end" in meta:
                lengths[i] = max(1, int(meta["period_end"]) - int(meta["period_begin"]) + 1)
            elif "obs_hours" in meta:
                lengths[i] = max(1, int(meta["obs_hours"]))

    # 2) from list_data style metadata (e.g., some datasets)
    if any(l <= 0 for l in lengths) and hasattr(ds, "list_data") and isinstance(ds.list_data, list) and len(ds.list_data) == n:
        for i, meta in enumerate(ds.list_data):
            if lengths[i] > 0 or not isinstance(meta, dict):
                continue
            if "table_length" in meta:
                lengths[i] = max(1, int(meta["table_length"]))

    # 3) from eager cached samples
    if any(l <= 0 for l in lengths) and hasattr(ds, "data") and isinstance(ds.data, list) and len(ds.data) == n:
        for i, item in enumerate(ds.data):
            if lengths[i] > 0 or not isinstance(item, dict):
                continue
            if "table_length" in item:
                lengths[i] = max(1, int(item["table_length"]))
            elif "measurement_table" in item and item["measurement_table"] is not None:
                try:
                    lengths[i] = max(1, len(item["measurement_table"]))
                except Exception:
                    pass

    # 4) optional fallback via __getitem__
    if fallback_from_getitem and any(l <= 0 for l in lengths):
        for i in range(n):
            if lengths[i] > 0:
                continue
            item = dataset[i]
            if isinstance(item, dict):
                if "table_length" in item:
                    lengths[i] = max(1, int(item["table_length"]))
                    continue
                if "measurement_table" in item and item["measurement_table"] is not None:
                    try:
                        lengths[i] = max(1, len(item["measurement_table"]))
                        continue
                    except Exception:
                        pass
            lengths[i] = 1

    # Final sanitize
    lengths = [l if isinstance(l, int) and l > 0 else 1 for l in lengths]
    return lengths


class SortishSampler(Sampler[int]):
    """
    Sortish sampler:
    - keeps length-similar samples nearby to reduce padding;
    - keeps randomness via chunk-level shuffling.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        chunk_factor: int = 50,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.chunk_factor = max(1, int(chunk_factor))
        self.seed = int(seed)

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        idxs = list(range(len(self.lengths)))
        if not self.shuffle:
            idxs.sort(key=lambda i: self.lengths[i], reverse=True)
            return iter(idxs)

        rng = random.Random(self.seed)
        rng.shuffle(idxs)

        chunk_size = self.batch_size * self.chunk_factor
        chunks = [idxs[i: i + chunk_size] for i in range(0, len(idxs), chunk_size)]
        chunks = [sorted(chunk, key=lambda i: self.lengths[i], reverse=True) for chunk in chunks]

        batches = []
        for chunk in chunks:
            for i in range(0, len(chunk), self.batch_size):
                batches.append(chunk[i: i + self.batch_size])

        if batches:
            max_len_batch_idx = max(range(len(batches)), key=lambda bi: self.lengths[batches[bi][0]])
            batches[0], batches[max_len_batch_idx] = batches[max_len_batch_idx], batches[0]
            if len(batches) > 1:
                head = batches[:1]
                tail = batches[1:]
                rng.shuffle(tail)
                batches = head + tail

        ordered = [i for b in batches for i in b]
        return iter(ordered)


class ApproxBatchSampler(Sampler[List[int]]):
    """
    Approximate token-batch sampler.
    Packs samples so that:
        max_seq_len_in_batch * batch_size <= max_tokens
    Also supports max_batch_size cap.
    """

    def __init__(
        self,
        sampler: Iterable[int],
        lengths: List[int],
        max_tokens: int,
        max_batch_size: Optional[int] = None,
        drop_last: bool = False,
    ):
        self.sampler = sampler
        self.lengths = lengths
        self.max_tokens = int(max_tokens)
        self.max_batch_size = None if max_batch_size is None else int(max_batch_size)
        self.drop_last = bool(drop_last)

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0 when provided")

    def __len__(self):
        # Approximate count for progress bars.
        if self.max_batch_size is None:
            return len(self.lengths)
        return (len(self.lengths) + self.max_batch_size - 1) // self.max_batch_size

    def __iter__(self):
        batch: List[int] = []
        cur_max_len = 0

        for idx in self.sampler:
            sample_len = max(1, int(self.lengths[idx]))
            next_max_len = max(cur_max_len, sample_len)
            next_bs = len(batch) + 1
            next_tokens = next_max_len * next_bs

            hit_token_cap = next_tokens > self.max_tokens
            hit_batch_cap = self.max_batch_size is not None and next_bs > self.max_batch_size

            if batch and (hit_token_cap or hit_batch_cap):
                if not self.drop_last or (self.max_batch_size is None or len(batch) == self.max_batch_size):
                    yield batch
                batch = [idx]
                cur_max_len = sample_len
            else:
                batch.append(idx)
                cur_max_len = next_max_len

        if batch:
            if not self.drop_last or (self.max_batch_size is None or len(batch) == self.max_batch_size):
                yield batch


def build_train_batch_sampler(
    dataset,
    per_device_batch_size: int,
    max_tokens_per_batch: int,
    use_sortish_sampler: bool = True,
    sortish_chunk_factor: int = 50,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = False,
):
    lengths = infer_sample_lengths(dataset)

    if use_sortish_sampler:
        base_sampler = SortishSampler(
            lengths=lengths,
            batch_size=per_device_batch_size,
            shuffle=shuffle,
            chunk_factor=sortish_chunk_factor,
            seed=seed,
        )
    else:
        base_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    return ApproxBatchSampler(
        sampler=base_sampler,
        lengths=lengths,
        max_tokens=max_tokens_per_batch,
        max_batch_size=per_device_batch_size,
        drop_last=drop_last,
    )


class TrainerWithBatchSampler(Trainer):
    """
    Trainer extension that supports custom train batch_sampler.
    """

    def __init__(self, *args, train_batch_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_batch_sampler = train_batch_sampler

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.train_batch_sampler is None:
            return super().get_train_dataloader()

        persistent_workers = (
            self.args.dataloader_persistent_workers and self.args.dataloader_num_workers > 0
        )
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=seed_worker,
        )
        return self.accelerator.prepare(train_dataloader)

