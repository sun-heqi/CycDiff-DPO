#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
从已生成的环肽构建偏好对
使用 XGBoost Ensemble 模型预测膜穿透性
"""

import sys
import os
import re
import pickle
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# 添加路径

from cyclicpeptide.Sequence2Structure import seq2stru_essentialAA

# ============ XGBoost Ensemble 预测器 ============
# (直接重建 ExtendedFeatureExtractor，避免 pickle 模块路径问题)

MODEL_DIR = './ckpts/xgboost_ensemble'

_xgb_models = None
_xgb_scaler = None
_xgb_extractor = None

# 重建 ExtendedFeatureExtractor（与 xgboost_train.py 中的逻辑一致）
XGB_RADIUS = 6
XGB_NBITS = 2048


def _make_extractor():
    """直接创建 ExtendedFeatureExtractor，不依赖 pickle"""
    sys.path.insert(0, './scripts')
    from xgboost_train import ExtendedFeatureExtractor
    return ExtendedFeatureExtractor(radius=XGB_RADIUS, nbits=XGB_NBITS, n_jobs=1)


def load_xgb_ensemble():
    """加载 XGBoost Ensemble 模型"""
    global _xgb_models, _xgb_scaler, _xgb_extractor

    if _xgb_models is None:
        # 加载 10 个模型
        model_names = ['base', 'more_trees', 'deeper', 'shallow', 'fast_lr',
                       'reg_strong', 'reg_l1', 'reg_l2', 'reg_balanced', 'conservative']
        _xgb_models = []
        for name in model_names:
            p = os.path.join(MODEL_DIR, f'model_{name}.pkl')
            with open(p, 'rb') as f:
                _xgb_models.append(pickle.load(f))
        print(f"✅ XGBoost Ensemble: 已加载 {len(_xgb_models)} 个模型")

        # 加载 scaler
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            _xgb_scaler = pickle.load(f)
        print(f"✅ Scaler 已加载")

        # 加载特征提取器（通过 _make_extractor 避免 pickle 模块路径问题）
        _xgb_extractor = _make_extractor()
        print(f"✅ Feature Extractor 已加载 (radius={XGB_RADIUS}, nbits={XGB_NBITS})")


def predict_permeability(smiles: str) -> float:
    """使用 XGBoost Ensemble 预测膜穿透性"""
    if _xgb_models is None:
        load_xgb_ensemble()

    features = _xgb_extractor.extract([smiles], silent=True)
    features_scaled = _xgb_scaler.transform(features)
    # 10 模型平均
    preds = np.array([m.predict(features_scaled)[0] for m in _xgb_models])
    return float(np.mean(preds))


def extract_sequence_from_pdb(pdb_file, chain_id):
    """从 PDB 文件提取指定链的序列"""
    from Bio.PDB import PDBParser

    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('pep', pdb_file)
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    seq = []
                    for residue in chain:
                        if residue.id[0] == ' ':
                            res_name = residue.resname
                            if res_name in aa_map:
                                seq.append(aa_map[res_name])
                    return ''.join(seq)
        return None
    except Exception:
        return None


def load_index_file(index_file):
    """
    加载 index.txt，返回:
      - sample_to_idx: {sample_name: idx}
      - sample_to_chain: {sample_name: peptide_chain}
    """
    sample_to_idx = {}
    sample_to_chain = {}
    with open(index_file, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            if parts:
                sample_name = parts[0]
                sample_to_idx[sample_name] = idx
                # 列7是peptide chain
                sample_to_chain[sample_name] = parts[7] if len(parts) > 7 else 'A'
    return sample_to_idx, sample_to_chain


def process_single_pdb(args):
    """处理单个 PDB 文件"""
    pdb_file, full_name, dataset_idx, sample_to_chain = args

    try:
        peptide_chain = sample_to_chain.get(full_name, 'A') if sample_to_chain else 'A'

        # 提取序列
        sequence = extract_sequence_from_pdb(str(pdb_file), peptide_chain)
        if sequence is None or len(sequence) == 0:
            return None

        # 序列 -> SMILES
        sequence = sequence.replace('X', 'A')
        smiles, _ = seq2stru_essentialAA(sequence=sequence, cyclic=True)

        # 预测膜穿透性
        permeability = predict_permeability(smiles)

        result = {
            'full_name': full_name,
            'pdb_file': str(pdb_file),
            'sequence': sequence,
            'smiles': smiles,
            'permeability_score': float(permeability),
            'length': len(sequence),
            'chain_id': peptide_chain,
            'dataset_idx': dataset_idx,
        }
        return result

    except Exception as e:
        return {'error': str(e), 'pdb_file': str(pdb_file)}


def process_generated_peptides(
    generated_dir,
    index_file,
    n_samples_per_target=None,
    output_scores='./generated_scores.pkl',
    output_pairs='./generated_pairs.pkl',
    min_score_diff=0.1,
    n_jobs=1
):
    """处理生成的环肽数据"""
    print("=" * 70)
    print("从生成数据构建偏好对 (XGBoost Ensemble)")
    print("=" * 70)

    # 加载 index.txt
    sample_to_idx = None
    sample_to_chain = None
    if index_file and os.path.exists(index_file):
        print(f"\n📋 加载数据集索引: {index_file}")
        sample_to_idx, sample_to_chain = load_index_file(index_file)
        print(f"   找到 {len(sample_to_idx)} 个有效样本")

    # 加载 XGBoost Ensemble 模型
    print("\n🔧 加载 XGBoost Ensemble 模型...")
    load_xgb_ensemble()

    # 扫描 PDB 文件
    print(f"\n🧬 扫描 PDB 文件...")
    generated_path = Path(generated_dir)
    all_pdb_files = list(generated_path.glob("*.pdb"))
    print(f"   找到 {len(all_pdb_files)} 个 PDB 文件")

    # 按靶标 ID 分组
    pdb_by_target = defaultdict(list)
    skipped = 0
    for pdb_file in all_pdb_files:
        full_name = pdb_file.stem
        if sample_to_idx and full_name not in sample_to_idx:
            skipped += 1
            continue

        # 提取靶标 ID: A_B_pdb1ddv_gen_68 -> A_B_pdb1ddv
        match = re.match(r'(.+)_gen_\d+', full_name)
        target_id = match.group(1) if match else full_name
        pdb_by_target[target_id].append(pdb_file)

    if skipped > 0:
        print(f"   ⚠️  跳过 {skipped} 个不在数据集中的样本")
    print(f"   分组到 {len(pdb_by_target)} 个靶标")

    # 准备任务
    all_tasks = []
    for target_id, file_list in pdb_by_target.items():
        if len(file_list) < 2:
            continue
        if n_samples_per_target and n_samples_per_target > 0 and len(file_list) > n_samples_per_target:
            file_list = random.sample(file_list, n_samples_per_target)
        for pdb_file in file_list:
            full_name = pdb_file.stem
            dataset_idx = sample_to_idx.get(full_name) if sample_to_idx else None
            all_tasks.append((pdb_file, full_name, dataset_idx, sample_to_chain))

    print(f"   准备处理 {len(all_tasks)} 个 PDB 文件 (n_jobs={n_jobs})")

    # 处理
    all_scores = {}
    target_peptides = defaultdict(list)
    errors = []
    count = 0

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(process_single_pdb, task): task for task in all_tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理 PDB"):
                result = future.result()
                if result is None:
                    continue
                if 'error' in result:
                    errors.append(result)
                    continue
                dataset_idx = result.get('dataset_idx', count)
                if result.get('dataset_idx') is None:
                    count += 1
                target_id_match = re.match(r'(.+)_gen_\d+', result['full_name'])
                target_id = target_id_match.group(1) if target_id_match else result['full_name']
                all_scores[dataset_idx] = {**result, 'target_id': target_id}
                target_peptides[target_id].append(dataset_idx)
    else:
        for task in tqdm(all_tasks, desc="处理 PDB"):
            result = process_single_pdb(task)
            if result is None:
                continue
            if 'error' in result:
                errors.append(result)
                continue
            dataset_idx = result.get('dataset_idx', count)
            if 'dataset_idx' not in result:
                count += 1
            target_id_match = re.match(r'(.+)_gen_\d+', result['full_name'])
            target_id = target_id_match.group(1) if target_id_match else result['full_name']
            all_scores[dataset_idx] = {**result, 'target_id': target_id}
            target_peptides[target_id].append(dataset_idx)

    if errors:
        print(f"\n⚠️  {len(errors)} 个错误，前5个:")
        for err in errors[:5]:
            print(f"   {err.get('pdb_file', '?')}: {err.get('error', '?')}")

    print(f"\n✅ 成功处理 {len(all_scores)} 个环肽，覆盖 {len(target_peptides)} 个靶标")

    # 保存分数
    output_scores = Path(output_scores)
    output_scores.parent.mkdir(parents=True, exist_ok=True)
    with open(output_scores, 'wb') as f:
        pickle.dump(all_scores, f)
    print(f"\n💾 分数已保存: {output_scores}")

    # 构建偏好对
    print(f"\n🔗 构建偏好对...")
    preference_pairs = {}
    pair_scores = {}

    for target_id, indices in target_peptides.items():
        if len(indices) < 2:
            continue
        sorted_indices = sorted(indices, key=lambda i: all_scores[i]['permeability_score'], reverse=True)
        n_pairs = len(sorted_indices) // 2
        for i in range(n_pairs):
            win_idx = sorted_indices[i]
            lose_idx = sorted_indices[-(i+1)]
            score_diff = all_scores[win_idx]['permeability_score'] - all_scores[lose_idx]['permeability_score']
            if score_diff >= min_score_diff:
                preference_pairs[win_idx] = lose_idx
                pair_scores[win_idx] = all_scores[win_idx]['permeability_score']
                pair_scores[lose_idx] = all_scores[lose_idx]['permeability_score']

    output_data = {
        'pairs': preference_pairs,
        'scores': pair_scores,
        'metadata': {
            'n_pairs': len(preference_pairs),
            'n_total_samples': len(all_scores),
            'n_targets': len(target_peptides),
            'min_score_diff': min_score_diff,
            'strategy': 'target_based',
            'predictor': 'xgboost_ensemble',
        }
    }

    with open(output_pairs, 'wb') as f:
        pickle.dump(output_data, f)

    if preference_pairs:
        score_diffs = [pair_scores[w] - pair_scores[preference_pairs[w]] for w in preference_pairs]
        print(f"\n✅ 构建了 {len(preference_pairs)} 个偏好对")
        print(f"   覆盖率: {len(preference_pairs)*2}/{len(all_scores)} "
              f"({len(preference_pairs)*2/len(all_scores)*100:.1f}%)")
        print(f"   分数差: min={min(score_diffs):.3f}, max={max(score_diffs):.3f}, "
              f"mean={np.mean(score_diffs):.3f}, std={np.std(score_diffs):.3f}")
    else:
        print(f"\n⚠️  警告: 没有构建任何偏好对")

    print(f"\n💾 偏好对已保存: {output_pairs}")
    return all_scores, output_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='使用 XGBoost Ensemble 从生成数据构建偏好对')
    parser.add_argument('--generated_dir', type=str,
                       default='./datasets/train_valid/pdbs',
                       help='生成数据目录')
    parser.add_argument('--cpsea_file', type=str,
                       default='./datasets/train_valid/processed/index.txt',
                       help='CPSea 靶标信息文件（已废弃，仅作占位）')
    parser.add_argument('--index_file', type=str,
                       default='./datasets/train_valid/processed/index.txt',
                       help='数据集索引文件 (processed/index.txt)')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='每个靶标随机选择的样本数 (None 表示使用所有)')
    parser.add_argument('--output_scores', type=str,
                       default='./datasets/train_valid/generated_scores.pkl',
                       help='输出分数文件')
    parser.add_argument('--output_pairs', type=str,
                       default='./datasets/train_valid/generated_pairs.pkl',
                       help='输出偏好对文件')
    parser.add_argument('--min_score_diff', type=float, default=0.1,
                       help='最小分数差')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--n_jobs', type=int, default=40,
                       help='并行进程数')

    args = parser.parse_args()

    if args.n_jobs < 1:
        args.n_jobs = 1

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n🎲 随机种子: {args.seed}")
    print(f"📁 PDB 目录: {args.generated_dir}")
    print(f"📋 数据集索引: {args.index_file}")
    print(f"📏 最小分数差: {args.min_score_diff}")
    print(f"🔧 并行进程数: {args.n_jobs}")

    scores, pairs = process_generated_peptides(
        generated_dir=args.generated_dir,
        index_file=args.index_file,
        n_samples_per_target=args.n_samples,
        output_scores=args.output_scores,
        output_pairs=args.output_pairs,
        min_score_diff=args.min_score_diff,
        n_jobs=args.n_jobs
    )

    print("\n🎉 完成！偏好对已保存，可用于 DPO 训练。")
