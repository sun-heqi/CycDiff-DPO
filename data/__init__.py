#!/usr/bin/python
# -*- coding:utf-8 -*-
from .dataset_wrapper import MixDatasetWrapper
from .codesign import CoDesignDataset
from .resample import ClusterResampler
from .dpo_dataset import create_dpo_dataset


import torch
from torch.utils.data import DataLoader

import utils.register as R
from utils.logger import print_log

def create_dataset(config: dict):
    # 检查是否启用DPO
    use_dpo = config.get('use_dpo', False)
    preference_pairs_path = config.get('preference_pairs_path', None)
    
    splits = []
    for split_name in ['train', 'valid', 'test']:
        split_config = config.get(split_name, None)
        if split_config is None:
            splits.append(None)
            continue
        if isinstance(split_config, list):
            dataset = MixDatasetWrapper(
                *[R.construct(cfg) for cfg in split_config]
            )
        else:
            dataset = R.construct(split_config)
        
        # 只对训练集应用DPO包装
        if split_name == 'train' and use_dpo and preference_pairs_path:
            print_log("=" * 60)
            print_log("🎯 启用DPO训练模式")
            print_log("=" * 60)
            dataset = create_dpo_dataset(
                dataset, 
                preference_pairs_path, 
                enabled=True
            )
        
        splits.append(dataset)
    return splits  # train/valid/test


def create_dataloader(dataset, config: dict, n_gpu: int=1, validation: bool=False):
    if 'wrapper' in config:
        dataset = R.construct(config['wrapper'], dataset=dataset)
    batch_size = config.get('batch_size', n_gpu) # default 1 on each gpu
    if validation:
        batch_size = config.get('val_batch_size', batch_size)
    shuffle = config.get('shuffle', False)
    num_workers = config.get('num_workers', 4)
    collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
    if n_gpu > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        batch_size = int(batch_size / n_gpu)
        print_log(f'Batch size on a single GPU: {batch_size}')
    else:
        sampler = None
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(shuffle and sampler is None),
        collate_fn=collate_fn,
        sampler=sampler
    )