#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost扩展特征提取器
"""

import os
import numpy as np
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# RDKit相关导入
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import MolFromSmiles
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.Descriptors import (
    TPSA, MolWt, MolLogP, NumValenceElectrons,
    MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge, MinAbsPartialCharge,
    BalabanJ, BertzCT, Chi0, Chi1, Chi0n, Chi1n,
    HallKierAlpha, Kappa1, Kappa2, Kappa3
)
from rdkit.Chem.Lipinski import (
    NumHDonors, NumHAcceptors, NumRotatableBonds,
    NumHeteroatoms, NumAliphaticRings, NumAromaticRings
)
from rdkit.Chem.rdMolDescriptors import (
    CalcNumRings, CalcFractionCSP3, CalcNumAmideBonds,
    CalcLabuteASA, CalcTPSA
)
from rdkit.Chem.EState.EState_VSA import (
    EState_VSA1, EState_VSA2, EState_VSA3
)


def extract_extended_features(smiles: str, radius: int, nbits: int) -> np.ndarray:
    """提取扩展特征集"""
    try:
        mol = MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nbits + 35)

        # ECFP指纹
        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        ecfp = np.zeros((nbits,))
        ConvertToNumpyArray(fp, ecfp)

        # 基础描述符
        tpsa = TPSA(mol)
        mw = MolWt(mol)
        clogp = MolLogP(mol)
        tpsa_mw_ratio = tpsa / mw if mw > 0 else 0.0

        # Lipinski描述符
        n_hdonors = NumHDonors(mol)
        n_hacceptors = NumHAcceptors(mol)
        n_rotatable = NumRotatableBonds(mol)
        n_heteroatoms = NumHeteroatoms(mol)
        n_aliphatic_rings = NumAliphaticRings(mol)
        n_aromatic_rings = NumAromaticRings(mol)

        # 原子相关
        heavy_atom_count = mol.GetNumHeavyAtoms()
        n_rings = CalcNumRings(mol)
        fraction_csp3 = CalcFractionCSP3(mol)

        # 拓扑描述符
        try:
            balaban_j = BalabanJ(mol)
        except:
            balaban_j = 0.0
        bertz_ct = BertzCT(mol)
        chi0 = Chi0(mol)
        chi1 = Chi1(mol)
        chi0n = Chi0n(mol)
        chi1n = Chi1n(mol)

        # Kappa形状指数
        try:
            hall_kier_alpha = HallKierAlpha(mol)
        except:
            hall_kier_alpha = 0.0
        kappa1 = Kappa1(mol)
        kappa2 = Kappa2(mol)
        kappa3 = Kappa3(mol)

        # 表面积
        labute_asa = CalcLabuteASA(mol)

        # 电荷相关
        try:
            max_partial_charge = MaxPartialCharge(mol)
            min_partial_charge = MinPartialCharge(mol)
            max_abs_partial_charge = MaxAbsPartialCharge(mol)
            min_abs_partial_charge = MinAbsPartialCharge(mol)
        except:
            max_partial_charge = 0.0
            min_partial_charge = 0.0
            max_abs_partial_charge = 0.0
            min_abs_partial_charge = 0.0

        # EState VSA
        estate_vsa1 = EState_VSA1(mol)
        estate_vsa2 = EState_VSA2(mol)
        estate_vsa3 = EState_VSA3(mol)

        # 其他
        num_valence_electrons = NumValenceElectrons(mol)
        try:
            n_amide_bonds = CalcNumAmideBonds(mol)
        except:
            n_amide_bonds = 0

        descriptors = np.array([
            tpsa, mw, tpsa_mw_ratio, clogp,
            n_hdonors, n_hacceptors, n_rotatable, n_heteroatoms,
            n_aliphatic_rings, n_aromatic_rings,
            heavy_atom_count, n_rings, fraction_csp3,
            balaban_j, bertz_ct, chi0, chi1, chi0n, chi1n,
            hall_kier_alpha, kappa1, kappa2, kappa3,
            labute_asa,
            max_partial_charge, min_partial_charge,
            max_abs_partial_charge, min_abs_partial_charge,
            estate_vsa1, estate_vsa2, estate_vsa3,
            num_valence_electrons, n_amide_bonds,
            n_hdonors + n_hacceptors,  # 总氢键数
            clogp / mw if mw > 0 else 0.0  # LogP/MW比
        ])

        return np.concatenate([ecfp, descriptors])
    except Exception:
        return np.zeros(nbits + 35)


class ExtendedFeatureExtractor:
    """扩展特征提取器"""

    def __init__(self, radius: int = 6, nbits: int = 1024, n_jobs: int = -1):
        self.radius = radius
        self.nbits = nbits
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.n_features = nbits + 35

    def extract(self, smiles_list: List[str], silent: bool = False) -> np.ndarray:
        """提取特征"""
        n_samples = len(smiles_list)
        features = np.zeros((n_samples, self.n_features))

        extract_func = partial(extract_extended_features,
                              radius=self.radius, nbits=self.nbits)

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(extract_func, smiles): i
                      for i, smiles in enumerate(smiles_list)}

            for future in tqdm(as_completed(futures), total=len(futures),
                             desc="Extracting features", disable=silent):
                idx = futures[future]
                features[idx] = future.result()

        return features
