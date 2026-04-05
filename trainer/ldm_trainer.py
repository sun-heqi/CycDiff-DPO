#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import pi, cos

import torch
from torch_scatter import scatter_mean

from .abs_trainer import Trainer
from utils import register as R


@R.register('LDMTrainer')
class LDMTrainer(Trainer):
    def __init__(self, model, train_loader, valid_loader, config: dict, save_config: dict, criterion: str='AAR'):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.max_step = self.config.max_epoch * len(self.train_loader)
        self.criterion = criterion
        assert criterion in ['AAR', 'RMSD', 'Loss'], f'Criterion {criterion} not implemented'
        self.rng_state = None
        
        # DPO training setup
        self.use_dpo = config.get('use_dpo', False)
        if self.use_dpo:
            dpo_beta = config.get('dpo_beta', 1000.0)
            dpo_loss_weight = config.get('dpo_loss_weight', 1.0)
            h_recon_weight = config.get('h_recon_weight', 1.0)
            x_recon_weight = config.get('x_recon_weight', 1.0)
            model.setup_dpo(dpo_beta, dpo_loss_weight, h_recon_weight, x_recon_weight)
            print(f"✅ DPO training enabled")
            print(f"   Loss weights: DPO={dpo_loss_weight}, H_recon={h_recon_weight}, X_recon={x_recon_weight}")

    ########## Override start ##########

    def train_step(self, batch, batch_idx):
        # Check if this is a DPO batch (has batch2 data)
        if self.use_dpo and self._is_dpo_batch(batch):
            # Split batch into winning and losing samples
            batch_w, batch_l = self._split_dpo_batch(batch)
            
            # Compute DPO loss
            loss, loss_dict = self.model.compute_dpo_loss(batch_w, batch_l)
            
            # Log DPO specific metrics
            self.log('DPO/Loss/Train', loss, batch_idx, val=False)
            
            if 'dpo_only' in loss_dict:
                self.log('DPO/DPO_Raw/Train', loss_dict['dpo_only'], batch_idx, val=False)
            if 'dpo_weighted' in loss_dict:
                self.log('DPO/DPO_Weighted/Train', loss_dict['dpo_weighted'], batch_idx, val=False)
            
            if 'h_recon_raw' in loss_dict:
                self.log('DPO/H_Recon_Raw/Train', loss_dict['h_recon_raw'], batch_idx, val=False)
            if 'h_recon_weighted' in loss_dict:
                self.log('DPO/H_Recon_Weighted/Train', loss_dict['h_recon_weighted'], batch_idx, val=False)
            
            if 'x_recon_raw' in loss_dict:
                self.log('DPO/X_Recon_Raw/Train', loss_dict['x_recon_raw'], batch_idx, val=False)
            if 'x_recon_weighted' in loss_dict:
                self.log('DPO/X_Recon_Weighted/Train', loss_dict['x_recon_weighted'], batch_idx, val=False)
            
            # 记录权重（用于监控）
            if 'dpo_weight' in loss_dict:
                self.log('DPO/DPO_Weight/Train', loss_dict['dpo_weight'], batch_idx, val=False)
            if 'h_recon_weight' in loss_dict:
                self.log('DPO/H_Recon_Weight/Train', loss_dict['h_recon_weight'], batch_idx, val=False)
            if 'x_recon_weight' in loss_dict:
                self.log('DPO/X_Recon_Weight/Train', loss_dict['x_recon_weight'], batch_idx, val=False)
            
            # 兼容旧代码的logging
            if 'reconstruction' in loss_dict:
                self.log('DPO/Reconstruction/Train', loss_dict['reconstruction'], batch_idx, val=False)
            
            self.log('DPO/Loss_W/Train', loss_dict['loss_w'], batch_idx, val=False)
            self.log('DPO/Loss_L/Train', loss_dict['loss_l'], batch_idx, val=False)
            
            # Log loss ratio (loss_w / loss_l)
            # Ideally, loss_w should be lower than loss_l (ratio < 1.0)
            loss_ratio = loss_dict['loss_w'] / (loss_dict['loss_l'] + 1e-8)
            self.log('DPO/Loss_Ratio_W_L/Train', loss_ratio, batch_idx, val=False)
            
            # Log reference model losses and ratios
            if 'loss_w_ref' in loss_dict and 'loss_l_ref' in loss_dict:
                self.log('DPO/Loss_W_Ref/Train', loss_dict['loss_w_ref'], batch_idx, val=False)
                self.log('DPO/Loss_L_Ref/Train', loss_dict['loss_l_ref'], batch_idx, val=False)
                loss_ratio_ref = loss_dict['loss_w_ref'] / (loss_dict['loss_l_ref'] + 1e-8)
                self.log('DPO/Loss_Ratio_W_L_Ref/Train', loss_ratio_ref, batch_idx, val=False)
            
            if 'dpo_H' in loss_dict:
                self.log('DPO/Loss_H/Train', loss_dict['dpo_H'], batch_idx, val=False)
                # Log component-wise ratio for sequence (H)
                if 'loss_w_H' in loss_dict and 'loss_l_H' in loss_dict:
                    loss_ratio_H = loss_dict['loss_w_H'] / (loss_dict['loss_l_H'] + 1e-8)
                    self.log('DPO/Loss_Ratio_H/Train', loss_ratio_H, batch_idx, val=False)
                    
            if 'dpo_X' in loss_dict:
                self.log('DPO/Loss_X/Train', loss_dict['dpo_X'], batch_idx, val=False)
                # Log component-wise ratio for structure (X)
                if 'loss_w_X' in loss_dict and 'loss_l_X' in loss_dict:
                    loss_ratio_X = loss_dict['loss_w_X'] / (loss_dict['loss_l_X'] + 1e-8)
                    self.log('DPO/Loss_Ratio_X/Train', loss_ratio_X, batch_idx, val=False)
        else:
            # Standard training
            results = self.model(**batch)
            if self.is_oom_return(results):
                return results
            loss, loss_dict = results

            self.log('Overall/Loss/Train', loss, batch_idx, val=False)

            if 'H' in loss_dict:
                self.log('Seq/Loss_H/Train', loss_dict['H'], batch_idx, val=False)

            if 'X' in loss_dict:
                self.log('Struct/Loss_X/Train', loss_dict['X'], batch_idx, val=False)

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log('lr', lr, batch_idx, val=False)

        return loss
    
    def _is_dpo_batch(self, batch):
        """Check if batch contains DPO pairs (has X2, S2, etc.)"""
        return 'X2' in batch or 'S2' in batch
    
    def _split_dpo_batch(self, batch):
        """Split DPO batch into winning and losing samples"""
        batch_w = {k: v for k, v in batch.items() if not k.endswith('2')}
        batch_l = {k.rstrip('2'): v for k, v in batch.items() if k.endswith('2')}
        return batch_w, batch_l

    def _valid_epoch_begin(self, device):
        self.rng_state = torch.random.get_rng_state()
        torch.manual_seed(12) # each validation epoch uses the same initial state
        return super()._valid_epoch_begin(device)

    def _valid_epoch_end(self, device):
        torch.random.set_rng_state(self.rng_state)
        return super()._valid_epoch_end(device)

    def valid_step(self, batch, batch_idx):
        loss, loss_dict = self.model(**batch)
        self.log('Overall/Loss/Validation', loss, batch_idx, val=True)
        if 'H' in loss_dict: self.log('Seq/Loss_H/Validation', loss_dict['H'], batch_idx, val=True)
        if 'X' in loss_dict: self.log('Struct/Loss_X/Validation', loss_dict['X'], batch_idx, val=True)
        # disable sidechain optimization as it may stuck for early validations where the model is still weak
        if self.local_rank != -1:  # ddp
            sample_X, sample_S, _ = self.model.module.sample(**batch, return_tensor=True, optimize_sidechain=False)
        else:
            sample_X, sample_S, _ = self.model.sample(**batch, return_tensor=True, optimize_sidechain=False)
        mask_generate = batch['mask']
        # batch ids
        batch_ids = torch.zeros_like(mask_generate).long()
        batch_ids[torch.cumsum(batch['lengths'], dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)
        batch_ids = batch_ids[mask_generate]

        if sample_S is not None:
            # aar
            aar = (batch['S'][mask_generate] == sample_S).float()
            aar = torch.mean(scatter_mean(aar, batch_ids, dim=-1))
            self.log('Seq/AAR/Validation', aar, batch_idx, val=True)

        # ca rmsd
        if sample_X is not None:
            atom_mask = batch['atom_mask'][mask_generate][:, 1]
            rmsd = ((batch['X'][mask_generate][:, 1][atom_mask] - sample_X[:, 1][atom_mask]) ** 2).sum(-1)  # [Ntgt]
            rmsd = torch.sqrt(scatter_mean(rmsd, batch_ids[atom_mask], dim=-1))  # [bs]
            rmsd = torch.mean(rmsd)

            self.log('Struct/CA_RMSD/Validation', rmsd, batch_idx, val=True)

        if self.criterion == 'AAR':
            return aar.detach()
        elif self.criterion == 'RMSD':
            return rmsd.detach()
        elif self.criterion == 'Loss':
            return loss.detach()
        else:
            raise NotImplementedError(f'Criterion {self.criterion} not implemented')

    def _train_epoch_end(self, device):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'update_epoch'):
            dataset.update_epoch()
        return super()._train_epoch_end(device)

    ########## Override end ##########