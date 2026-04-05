#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
DPO数据集包装器
将偏好对数据包装成训练batch
"""

import pickle
import torch
from torch.utils.data import Dataset


class DPODatasetWrapper(Dataset):
    """
    DPO数据集包装器
    将原始数据集 + 偏好对索引 转换为DPO训练数据
    """
    
    def __init__(self, original_dataset, preference_pairs_path):
        """
        Args:
            original_dataset: CP-Composer的原始数据集
            preference_pairs_path: 偏好对pkl文件路径
        """
        self.original_dataset = original_dataset
        
        # 加载偏好对
        print(f"📦 加载偏好对: {preference_pairs_path}")
        with open(preference_pairs_path, 'rb') as f:
            data = pickle.load(f)
        
        self.pairs = data['pairs']  # {winning_idx: losing_idx}
        self.scores = data.get('scores', {})
        self.valid_indices = list(self.pairs.keys())
        
        print(f"✅ DPO数据集: {len(self.valid_indices)} 个偏好对")
        if 'metadata' in data:
            print(f"   策略: {data['metadata'].get('strategy', 'unknown')}")
            print(f"   覆盖率: {data['metadata'].get('n_pairs', 0)*2}/{data['metadata'].get('n_total_samples', 0)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        返回一个DPO训练样本（winning + losing）
        """
        # 获取winning和losing的索引
        win_idx = self.valid_indices[idx]
        lose_idx = self.pairs[win_idx]
        
        # 从原始数据集加载
        win_sample = self.original_dataset[win_idx]
        lose_sample = self.original_dataset[lose_idx]
        
        # 构建DPO batch
        # winning sample的字段保持原样
        dpo_sample = {}
        for k, v in win_sample.items():
            dpo_sample[k] = v
        
        # losing sample的字段添加'2'后缀
        for k, v in lose_sample.items():
            dpo_sample[k + '2'] = v
        
        # 注意：不添加标量分数字段，避免collate_fn错误
        # 如果需要记录分数，可以在训练时从pkl文件读取
        
        return dpo_sample
    
    def get_summary(self, idx):
        """兼容原数据集的接口"""
        win_idx = self.valid_indices[idx]
        if hasattr(self.original_dataset, 'get_summary'):
            return self.original_dataset.get_summary(win_idx)
        return None
    
    def get_len(self, idx):
        """返回样本长度（用于DynamicBatchWrapper）"""
        win_idx = self.valid_indices[idx]
        if hasattr(self.original_dataset, 'get_len'):
            return self.original_dataset.get_len(win_idx)
        # 如果没有get_len方法，尝试直接获取样本并返回长度
        try:
            sample = self.original_dataset[win_idx]
            if 'lengths' in sample:
                return sample['lengths'].item() if hasattr(sample['lengths'], 'item') else sample['lengths']
            elif 'mask' in sample:
                return sample['mask'].sum().item()
            else:
                return 10  # 默认长度
        except:
            return 10  # 默认长度
    
    @property
    def collate_fn(self):
        """代理到原始数据集的collate_fn"""
        if hasattr(self.original_dataset, 'collate_fn'):
            return self.original_dataset.collate_fn
        return None


def create_dpo_dataset(original_dataset, preference_pairs_path, enabled=True):
    """
    创建DPO数据集
    
    Args:
        original_dataset: 原始数据集
        preference_pairs_path: 偏好对文件路径
        enabled: 是否启用DPO（如果False则返回原数据集）
    
    Returns:
        DPO数据集或原数据集
    """
    if not enabled:
        print("⚠️  DPO未启用，使用标准数据集")
        return original_dataset
    
    try:
        dpo_dataset = DPODatasetWrapper(original_dataset, preference_pairs_path)
        return dpo_dataset
    except Exception as e:
        print(f"❌ 创建DPO数据集失败: {e}")
        print("⚠️  回退到标准数据集")
        return original_dataset

