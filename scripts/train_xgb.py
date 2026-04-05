#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
XGBoost Ensemble Training Script
Trains an ensemble of XGBoost models for caco2 permeability prediction.
"""
import sys, os, json, time
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import pickle


def main():
    output_dir = Path(__file__).parent.parent / 'ckpts' / 'xgboost_ensemble'
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file = Path(__file__).parent.parent / 'datasets' / 'caco2' / 'caco2_dedup.csv'

    configs = [
        {'name': 'base', 'seed': 42, 'n_estimators': 210, 'max_depth': 7, 'lr': 0.08, 'subsample': 0.74, 'colsample': 0.74, 'reg_alpha': 0.01, 'reg_lambda': 1.0},
        {'name': 'more_trees', 'seed': 123, 'n_estimators': 250, 'max_depth': 7, 'lr': 0.07, 'subsample': 0.76, 'colsample': 0.76, 'reg_alpha': 0.01, 'reg_lambda': 1.0},
        {'name': 'deeper', 'seed': 456, 'n_estimators': 180, 'max_depth': 9, 'lr': 0.08, 'subsample': 0.72, 'colsample': 0.72, 'reg_alpha': 0.01, 'reg_lambda': 1.0},
        {'name': 'shallow', 'seed': 789, 'n_estimators': 280, 'max_depth': 5, 'lr': 0.06, 'subsample': 0.78, 'colsample': 0.78, 'reg_alpha': 0.01, 'reg_lambda': 1.0},
        {'name': 'fast_lr', 'seed': 999, 'n_estimators': 150, 'max_depth': 7, 'lr': 0.1, 'subsample': 0.74, 'colsample': 0.74, 'reg_alpha': 0.01, 'reg_lambda': 1.0},
        {'name': 'reg_strong', 'seed': 111, 'n_estimators': 250, 'max_depth': 6, 'lr': 0.07, 'subsample': 0.75, 'colsample': 0.75, 'reg_alpha': 0.1, 'reg_lambda': 1.5},
        {'name': 'reg_l1', 'seed': 222, 'n_estimators': 230, 'max_depth': 7, 'lr': 0.075, 'subsample': 0.74, 'colsample': 0.74, 'reg_alpha': 0.2, 'reg_lambda': 1.0},
        {'name': 'reg_l2', 'seed': 333, 'n_estimators': 230, 'max_depth': 7, 'lr': 0.075, 'subsample': 0.74, 'colsample': 0.74, 'reg_alpha': 0.05, 'reg_lambda': 2.0},
        {'name': 'reg_balanced', 'seed': 444, 'n_estimators': 240, 'max_depth': 6, 'lr': 0.07, 'subsample': 0.76, 'colsample': 0.76, 'reg_alpha': 0.08, 'reg_lambda': 1.3},
        {'name': 'conservative', 'seed': 555, 'n_estimators': 300, 'max_depth': 5, 'lr': 0.05, 'subsample': 0.78, 'colsample': 0.78, 'reg_alpha': 0.05, 'reg_lambda': 1.2},
    ]

    print(f'加载数据: {data_file}')
    df_all = pd.read_csv(data_file)
    smiles_list = df_all['smiles'].tolist()
    y_values = df_all['caco2'].values
    print(f'  样本数: {len(y_values)}')

    sys.path.insert(0, str(Path(__file__).parent))
    from xgboost_train import ExtendedFeatureExtractor

    print('提取特征...')
    extractor = ExtendedFeatureExtractor(radius=6, nbits=2048, n_jobs=40)
    X = extractor.extract(smiles_list, silent=False)
    print(f'  维度: {X.shape}')

    print('标准化...')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open(output_dir / 'extractor.pkl', 'wb') as f:
        pickle.dump(extractor, f)

    print('=' * 60)
    print(f'训练 {len(configs)} 个模型...')

    models = []
    for i, cfg in enumerate(configs):
        print(f'\n[{i+1}/{len(configs)}] {cfg["name"]}')
        model = xgb.XGBRegressor(
            n_estimators=cfg['n_estimators'], max_depth=cfg['max_depth'],
            learning_rate=cfg['lr'], subsample=cfg['subsample'],
            colsample_bytree=cfg['colsample'], reg_alpha=cfg['reg_alpha'],
            reg_lambda=cfg['reg_lambda'], min_child_weight=2,
            random_state=cfg['seed'], n_jobs=-1,
            objective='reg:squarederror', device='cpu'
        )
        model.fit(X_scaled, y_values)
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y_values, y_pred)
        pearson, _ = pearsonr(y_values, y_pred)
        print(f'    R²={r2:.4f}, Pearson={pearson:.4f}')
        with open(output_dir / f'model_{cfg["name"]}.pkl', 'wb') as f:
            pickle.dump(model, f)
        models.append(model)

    y_avg = np.mean([m.predict(X_scaled) for m in models], axis=0)
    avg_r2 = r2_score(y_values, y_avg)
    avg_pearson, _ = pearsonr(y_values, y_avg)
    print(f'\n集成: R²={avg_r2:.4f}, Pearson={avg_pearson:.4f}')

    config_data = {
        'n_models': len(configs),
        'ensemble_configs': configs,
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(y_values),
        'feature_dim': X.shape[1],
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f'\n模型已保存至: {output_dir}')
    print('完成!')


if __name__ == '__main__':
    main()
