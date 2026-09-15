import collections
import itertools
import copy
import math

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from torch.utils.data.sampler import Sampler

# https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/core/utils/dist_utils.py#L157
def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed.
    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.
    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


class GroupEachSampleInBatchSamplerIterWeighted(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.
    Shuffling is only done for group order, not done within groups.
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 group_weights=None,
                 seed=0):

        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)
        # MMCV's DistSamplerSeedHook expects batch_sampler.sampler.set_epoch().
        self.sampler = self
        self.group_weights = group_weights if group_weights is not None else np.ones(
            len(np.bincount(self.dataset.flag)))

        self.size = len(self.dataset)

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = batch_size * world_size

        assert self.groups_num >= self.global_batch_size

        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)}

        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx)
            for local_sample_idx in range(self.batch_size)]

        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]

    def _infinite_group_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.multinomial(torch.tensor(self.group_weights, dtype=torch.float32), self.groups_num,
                                         replacement=True, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(self._infinite_group_indices(),
                                    global_sample_idx,
                                    None,
                                    self.global_batch_size)

    def __iter__(self):
        while True:
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                    self.buffer_per_local_sample[local_sample_idx] = \
                        copy.deepcopy(self.group_idx_to_sample_idxs[new_group_idx])

                curr_batch.append(self.buffer_per_local_sample[local_sample_idx].pop(0))

            yield curr_batch

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        self.epoch = epoch

class GroupEachSampleInBatchSamplerIter(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.
    Shuffling is only done for group order, not done within groups.
    """

    def __init__(self, 
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0):

        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)
        # MMCV's DistSamplerSeedHook expects batch_sampler.sampler.set_epoch().
        self.sampler = self

        self.size = len(self.dataset)

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = batch_size * world_size

        assert self.groups_num >= self.global_batch_size

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)}        

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator 
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx) 
            for local_sample_idx in range(self.batch_size)]
        
        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]

    def _infinite_group_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(self._infinite_group_indices(), 
                                    global_sample_idx, 
                                    None,
                                    self.global_batch_size)

    def __iter__(self):
        while True:
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                    self.buffer_per_local_sample[local_sample_idx] = \
                        copy.deepcopy(
                            self.group_idx_to_sample_idxs[new_group_idx])

                curr_batch.append(self.buffer_per_local_sample[local_sample_idx].pop(0))
            
            yield curr_batch

    def __len__(self):
        """Length of base dataset."""
        return self.size
        
    def set_epoch(self, epoch):
        self.epoch = epoch


class GroupEachSampleInBatchSamplerIterEval(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.
    Shuffling is only done for group order, not done within groups.
    """

    def __init__(self, 
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0):

        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)
        # MMCV's DistSamplerSeedHook expects batch_sampler.sampler.set_epoch().
        self.sampler = self

        self.size = len(self.dataset)

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = batch_size * world_size

        assert self.groups_num >= self.global_batch_size

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)}        

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator 
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx) 
            for local_sample_idx in range(self.batch_size)]
        
        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]

    def _infinite_group_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(self._infinite_group_indices(), 
                                    global_sample_idx, 
                                    None,
                                    self.global_batch_size)

    def __iter__(self):

        t = (len(self.flag)+self.world_size*16 + 1)//self.world_size
        for i in range(t//self.batch_size):
            if i == 0: self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                    self.buffer_per_local_sample[local_sample_idx] = \
                        copy.deepcopy(
                            self.group_idx_to_sample_idxs[new_group_idx])

                curr_batch.append(self.buffer_per_local_sample[local_sample_idx].pop(0))
            
            yield curr_batch

    def __len__(self):
        """Length of base dataset."""
        return self.size
        
    def set_epoch(self, epoch):
        self.epoch = epoch


class GroupEachSampleInBatchSamplerEpoch(Sampler):
    """Epoch-based variant of group-wise sequential sampler.

    It keeps sequence continuity within each batch slot:
    group order is shuffled, but sample order inside each group is preserved.
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0,
                 drop_last=False):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)
        # MMCV's DistSamplerSeedHook expects batch_sampler.sampler.set_epoch().
        self.sampler = self
        self.drop_last = drop_last
        self.epoch = 0

        self.size = len(self.dataset)
        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = batch_size * world_size
        assert self.groups_num >= self.global_batch_size

        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)
        }

        # Assign every sequence exactly once to a global batch slot.  Longest
        # first greedy packing keeps slot lengths balanced while preserving the
        # chronological order within each sequence.
        tie_generator = torch.Generator()
        tie_generator.manual_seed(self.seed)
        tie_order = torch.randperm(
            self.groups_num, generator=tie_generator).tolist()
        group_order = sorted(
            tie_order, key=lambda group_idx: -self.group_sizes[group_idx])
        self.slot_group_indices = [
            [] for _ in range(self.global_batch_size)]
        self.slot_lengths = [0 for _ in range(self.global_batch_size)]
        for group_idx in group_order:
            slot_idx = min(
                range(self.global_batch_size),
                key=lambda index: (self.slot_lengths[index], index))
            self.slot_group_indices[slot_idx].append(group_idx)
            self.slot_lengths[slot_idx] += len(
                self.group_idx_to_sample_idxs[group_idx])

        # LPT packing is close to optimal but can leave an avoidable gap when
        # two groups of slightly different lengths can be exchanged.  Refine
        # the longest/shortest pair so padding does not duplicate whole tail
        # segments merely because of the initial greedy tie order.
        while self.slot_lengths:
            longest = int(np.argmax(self.slot_lengths))
            shortest = int(np.argmin(self.slot_lengths))
            current_range = (
                self.slot_lengths[longest] - self.slot_lengths[shortest])
            if current_range <= 1:
                break
            best_swap = None
            best_range = current_range
            best_pair_gap = current_range
            for long_pos, long_group in enumerate(
                    self.slot_group_indices[longest]):
                long_size = len(self.group_idx_to_sample_idxs[long_group])
                for short_pos, short_group in enumerate(
                        self.slot_group_indices[shortest]):
                    short_size = len(
                        self.group_idx_to_sample_idxs[short_group])
                    if long_size <= short_size:
                        continue
                    candidate_lengths = list(self.slot_lengths)
                    delta = long_size - short_size
                    candidate_lengths[longest] -= delta
                    candidate_lengths[shortest] += delta
                    candidate_range = max(candidate_lengths) - min(
                        candidate_lengths)
                    candidate_pair_gap = abs(
                        candidate_lengths[longest] -
                        candidate_lengths[shortest])
                    if (candidate_range < best_range or
                            (candidate_range == best_range and
                             candidate_pair_gap < best_pair_gap)):
                        best_range = candidate_range
                        best_pair_gap = candidate_pair_gap
                        best_swap = (
                            long_pos, short_pos, candidate_lengths)
            if best_swap is None:
                break
            long_pos, short_pos, candidate_lengths = best_swap
            (self.slot_group_indices[longest][long_pos],
             self.slot_group_indices[shortest][short_pos]) = (
                self.slot_group_indices[shortest][short_pos],
                self.slot_group_indices[longest][long_pos])
            self.slot_lengths = candidate_lengths

        if self.drop_last:
            self.num_batches = min(self.slot_lengths, default=0)
        else:
            # Shorter slots repeat only their final sample at the tail.  Unlike
            # the old fixed ceil(size/global_batch) loop, no real sample is
            # truncated or silently omitted from an epoch.
            self.num_batches = max(self.slot_lengths, default=0)

    def _infinite_group_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        while True:
            yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(
            self._infinite_group_indices(),
            global_sample_idx,
            None,
            self.global_batch_size)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        partition_order = torch.randperm(
            self.global_batch_size, generator=generator).tolist()
        slot_sample_sequences = []
        global_slot_start = self.rank * self.batch_size
        global_slot_end = global_slot_start + self.batch_size
        for global_slot_idx in range(global_slot_start, global_slot_end):
            partition_idx = partition_order[global_slot_idx]
            groups = self.slot_group_indices[partition_idx]
            if groups:
                group_permutation = torch.randperm(
                    len(groups), generator=generator).tolist()
                groups = [groups[index] for index in group_permutation]
            samples = []
            for group_idx in groups:
                samples.extend(self.group_idx_to_sample_idxs[group_idx])
            slot_sample_sequences.append(samples)

        slot_positions = [0 for _ in range(self.batch_size)]
        slot_pad_values = [
            sequence[-1] if sequence else None
            for sequence in slot_sample_sequences]
        for _ in range(self.num_batches):
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                sequence = slot_sample_sequences[local_sample_idx]
                position = slot_positions[local_sample_idx]
                if position < len(sequence):
                    sample_idx = sequence[position]
                    slot_positions[local_sample_idx] += 1
                    slot_pad_values[local_sample_idx] = sample_idx
                else:
                    sample_idx = slot_pad_values[local_sample_idx]
                    if sample_idx is None:
                        raise RuntimeError(
                            'Temporal training sampler produced an empty slot.')
                curr_batch.append(sample_idx)
            yield curr_batch

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch


class GroupEachSampleInBatchSamplerEpochWeighted(GroupEachSampleInBatchSamplerEpoch):
    """Epoch-based sequence sampler with weighted group selection.

    This preserves temporal order inside each selected group, but samples group
    ids with replacement according to per-group weights. It is intended for
    sequence-level class balancing where frame-level shuffling would break
    temporal caches.
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0,
                 group_weights=None,
                 drop_last=False):
        super().__init__(
            dataset,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            seed=seed,
            drop_last=drop_last)
        if group_weights is None:
            group_weights = np.ones(self.groups_num, dtype=np.float32)
        group_weights = np.asarray(group_weights, dtype=np.float32)
        if group_weights.shape[0] != self.groups_num:
            raise ValueError(
                'group_weights length must match number of sequence groups: '
                f'{group_weights.shape[0]} vs {self.groups_num}.')
        group_weights = np.maximum(group_weights, 1e-6)
        self.group_weights = group_weights
        # Weighted sampling intentionally draws groups with replacement for a
        # conventional dataset-sized epoch; it does not use the complete-pass
        # partition schedule from the base class.
        self.num_batches = (
            self.size // self.global_batch_size if self.drop_last else
            math.ceil(self.size / self.global_batch_size))

    def __iter__(self):
        buffer_per_local_sample = [[] for _ in range(self.batch_size)]
        group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(
                self.rank * self.batch_size + local_sample_idx)
            for local_sample_idx in range(self.batch_size)
        ]
        for _ in range(self.num_batches):
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if not buffer_per_local_sample[local_sample_idx]:
                    group_idx = next(
                        group_indices_per_global_sample_idx[local_sample_idx])
                    buffer_per_local_sample[local_sample_idx] = copy.deepcopy(
                        self.group_idx_to_sample_idxs[group_idx])
                curr_batch.append(
                    buffer_per_local_sample[local_sample_idx].pop(0))
            yield curr_batch

    def _infinite_group_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        weights = torch.tensor(self.group_weights, dtype=torch.float32)
        while True:
            yield from torch.multinomial(
                weights,
                self.groups_num,
                replacement=True,
                generator=g).tolist()


class GroupEachSampleInBatchSamplerEpochEval(Sampler):
    """Epoch-based eval sampler with sequence continuity.

    It is finite and deterministic by default (no group shuffle), and
    preserves sample order inside each group.
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0,
                 shuffle=False):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)
        # MMCV's DistSamplerSeedHook expects batch_sampler.sampler.set_epoch().
        self.sampler = self
        self.shuffle = shuffle
        self.epoch = 0

        self.size = len(self.dataset)
        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = batch_size * world_size

        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)
        }
        self.num_batches = self._num_batches_for_group_order(self._group_order())

    def _global_slot_lengths(self, group_order):
        slot_lengths = []
        for global_slot_idx in range(self.global_batch_size):
            slot_length = 0
            for group_idx in group_order[global_slot_idx::self.global_batch_size]:
                slot_length += len(self.group_idx_to_sample_idxs[group_idx])
            slot_lengths.append(slot_length)
        return slot_lengths

    def _num_batches_for_group_order(self, group_order):
        # Eval keeps per-slot sequence continuity. Different slots therefore
        # have different sequence lengths, so using ceil(size / global_batch)
        # drops the tail of the longest slots. Run until the longest slot is
        # exhausted and let shorter slots pad with their last valid sample.
        slot_lengths = self._global_slot_lengths(group_order)
        return max(slot_lengths, default=0)

    def _group_order(self):
        if not self.shuffle:
            order = torch.arange(self.groups_num).tolist()
        else:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.groups_num, generator=g).tolist()

        # Small smoke subsets may contain fewer groups than the distributed
        # global batch size. Repeat the deterministic group order so every
        # rank/slot still gets an assigned sequence.
        if len(order) < self.global_batch_size:
            repeat_factor = math.ceil(self.global_batch_size / max(len(order), 1))
            order = (order * repeat_factor)[:self.global_batch_size]
        return order

    def __iter__(self):
        group_order = self._group_order()
        num_batches = self._num_batches_for_group_order(group_order)
        slot_sample_sequences = []
        global_slot_start = self.rank * self.batch_size
        global_slot_end = global_slot_start + self.batch_size

        for global_slot_idx in range(global_slot_start, global_slot_end):
            assigned_group_indices = group_order[global_slot_idx::self.global_batch_size]
            assigned_samples = []
            for group_idx in assigned_group_indices:
                assigned_samples.extend(self.group_idx_to_sample_idxs[group_idx])
            slot_sample_sequences.append(assigned_samples)

        slot_positions = [0 for _ in range(self.batch_size)]
        slot_pad_values = [
            sample_sequence[-1] if len(sample_sequence) > 0 else None
            for sample_sequence in slot_sample_sequences
        ]

        for _ in range(num_batches):
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                sample_sequence = slot_sample_sequences[local_sample_idx]
                sample_pos = slot_positions[local_sample_idx]
                if sample_pos < len(sample_sequence):
                    sample_idx = sample_sequence[sample_pos]
                    slot_positions[local_sample_idx] += 1
                    slot_pad_values[local_sample_idx] = sample_idx
                else:
                    sample_idx = slot_pad_values[local_sample_idx]
                    if sample_idx is None:
                        raise RuntimeError('Eval sampler produced an empty slot.')
                curr_batch.append(sample_idx)
            yield curr_batch

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.shuffle:
            self.num_batches = self._num_batches_for_group_order(
                self._group_order())


class TTADistributedSampler(Sampler):

    def __init__(self,
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank

        self.dataset = dataset
        assert batch_size == 1
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)
        # Keep the same interface as PyTorch BatchSampler wrappers.
        self.sampler = self

        self.size = len(self.dataset)

    def __iter__(self):
        indices = torch.arange(len(self.dataset)).tolist()
        for i in indices:
            yield [i]

    def __len__(self):
        """Length of base dataset."""
        return self.size * 8
