#!/usr/bin/python
# -*- coding:utf-8 -*-
import enum

import torch
import torch.nn as nn

import utils.register as R
from utils.oom_decorator import oom_decorator
from data.format import VOCAB

from .diffusion.dpm_full import FullDPM
from .energies.dist import dist_energy
from ..autoencoder.model import AutoEncoder


@R.register('LDMPepDesign')
class LDMPepDesign(nn.Module):

    def __init__(
            self,
            autoencoder_ckpt,
            autoencoder_no_randomness,
            hidden_size,
            num_steps,
            n_layers,
            dist_rbf=0,
            dist_rbf_cutoff=7.0,
            n_rbf=0,
            cutoff=1.0,
            max_gen_position=30,
            mode='codesign',
            h_loss_weight=None,
            diffusion_opt={}):
        super().__init__()
        self.autoencoder_no_randomness = autoencoder_no_randomness
        self.latent_idx = VOCAB.symbol_to_idx(VOCAB.LAT)

        self.autoencoder: AutoEncoder = torch.load(autoencoder_ckpt, map_location='cpu')
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        
        self.train_sequence, self.train_structure = True, True
        if mode == 'fixbb':
            self.train_structure = False
        elif mode == 'fixseq':
            self.train_sequence = False
        
        latent_size = self.autoencoder.latent_size if self.train_sequence else hidden_size

        self.abs_position_encoding = nn.Embedding(max_gen_position, latent_size)
        self.diffusion = FullDPM(
            latent_size=latent_size,
            hidden_size=hidden_size,
            n_channel=self.autoencoder.n_channel,
            num_steps=num_steps,
            n_layers=n_layers,
            n_rbf=n_rbf,
            cutoff=cutoff,
            dist_rbf=dist_rbf,
            dist_rbf_cutoff=dist_rbf_cutoff,
            **diffusion_opt
        )
        if self.train_sequence:
            self.hidden2latent = nn.Linear(hidden_size, self.autoencoder.latent_size)
            if h_loss_weight is None:
                self.h_loss_weight = self.autoencoder.latent_n_channel * 3 / self.autoencoder.latent_size  # make loss_X and loss_H about the same size
            else:
                self.h_loss_weight = h_loss_weight
        if self.train_structure:
            # for better constrained sampling
            self.consec_dist_mean, self.consec_dist_std = None, None

    @oom_decorator
    def forward(self, X, S, mask, position_ids, lengths, atom_mask, L=None, t=None):
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
            t: timestep for diffusion (optional, for DPO training)
        '''

        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            H, Z, _, _ = self.autoencoder.encode(X, S, mask, position_ids, lengths, atom_mask, no_randomness=self.autoencoder_no_randomness)
        
        # diffusion model
        if self.train_sequence:
            S = S.clone()
            S[mask] = self.latent_idx

        with torch.no_grad():
            H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        position_embedding = self.abs_position_encoding(torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))

        if self.train_sequence:
            H_0 = self.hidden2latent(H_0)
            H_0 = H_0.clone()
            H_0[mask] = H
        
        if self.train_structure:
            X = X.clone()
            X[mask] = self.autoencoder._fill_latent_channels(Z)
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, :self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel:] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:  # fixbb, only retain backbone atoms in masked region
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        loss_dict = self.diffusion.forward(
            H_0=H_0,
            X_0=X,
            position_embedding=position_embedding,
            mask_generate=mask,
            lengths=lengths,
            atom_embeddings=atom_embeddings,
            atom_mask=atom_mask,
            L=L,
            t=t,  # ⭐ 传递timestep
            sample_structure=self.train_structure,
            sample_sequence=self.train_sequence
        )
        

        # loss
        loss = 0
        if self.train_sequence:
            loss = loss + loss_dict['H'] * self.h_loss_weight
        if self.train_structure:
            loss = loss + loss_dict['X']

        return loss, loss_dict

    def set_consec_dist(self, mean: float, std: float):
        self.consec_dist_mean = mean
        self.consec_dist_std = std

    def latent_geometry_guidance(self, X, mask_generate, batch_ids, tolerance=3, **kwargs):
        assert self.consec_dist_mean is not None and self.consec_dist_std is not None, \
               'Please run set_consec_dist(self, mean, std) to setup guidance parameters'
        return dist_energy(
            X, mask_generate, batch_ids,
            self.consec_dist_mean, self.consec_dist_std,
            tolerance=tolerance, **kwargs
        )
    

    @torch.no_grad()
    def sample(
        self,
        X, S, mask, position_ids, lengths, atom_mask, L=None,
        sample_opt={
            'pbar': False,
            'energy_func': None,
            'energy_lambda': 0.0,
            'autoencoder_n_iter': 1
        },
        return_tensor=False,
        optimize_sidechain=True,
    ):
        self.autoencoder.eval()
        # diffusion sample
        if self.train_sequence:
            S = S.clone()
            S[mask] = self.latent_idx

        H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        position_embedding = self.abs_position_encoding(torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))

        if self.train_sequence:
            H_0 = self.hidden2latent(H_0)
            H_0 = H_0.clone()
            H_0[mask] = 0 # no possibility for leakage

        if self.train_structure:
            X = X.clone()
            X[mask] = 0
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, :self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel:] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:  # fixbb, only retain backbone atoms in masked region
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        sample_opt['sample_sequence'] = self.train_sequence
        sample_opt['sample_structure'] = self.train_structure
        if 'energy_func' in sample_opt:
            if sample_opt['energy_func'] is None:
                pass
            elif sample_opt['energy_func'] == 'default':
                sample_opt['energy_func'] = self.latent_geometry_guidance
            # otherwise this should be a function
        autoencoder_n_iter = sample_opt.pop('autoencoder_n_iter', 1)
        
        traj = self.diffusion.sample(H_0, X, position_embedding, mask, lengths, atom_embeddings, atom_mask, L, **sample_opt)
        X_0, H_0 = traj[0]
        X_0, H_0 = X_0[mask][:, :self.autoencoder.latent_n_channel], H_0[mask]

        # autodecoder decode
        batch_X, batch_S, batch_ppls = self.autoencoder.test(
            X, S, mask, position_ids, lengths, atom_mask,
            given_laten_H=H_0, given_latent_X=X_0, return_tensor=return_tensor,
            allow_unk=False, optimize_sidechain=optimize_sidechain,
            n_iter=autoencoder_n_iter
        )

        return batch_X, batch_S, batch_ppls
    
    def setup_dpo(self, dpo_beta=1000.0, dpo_loss_weight=1.0, h_recon_weight=1.0, x_recon_weight=1.0):
        """Setup DPO training by creating a reference model
        
        Args:
            dpo_beta: DPO temperature parameter (controls preference strength)
            dpo_loss_weight: Weight for DPO loss (控制DPO loss的权重)
            h_recon_weight: Weight for sequence reconstruction loss (序列重建loss权重)
            x_recon_weight: Weight for structure reconstruction loss (结构重建loss权重)
        """
        import copy
        self.dpo_beta = dpo_beta
        self.dpo_loss_weight = dpo_loss_weight
        self.h_recon_weight = h_recon_weight
        self.x_recon_weight = x_recon_weight
        
        # Create reference model (frozen deep copy)
        self.ref_model = copy.deepcopy(self)
        self.ref_model.eval()
        
        # Freeze all parameters
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        total_weight = dpo_loss_weight + h_recon_weight + x_recon_weight
        print(f"✅ DPO setup complete with beta={dpo_beta}")
        print(f"   Loss weights (normalized percentages):")
        print(f"     - DPO loss:      {dpo_loss_weight:6.2f} ({dpo_loss_weight/total_weight*100:5.1f}%)")
        print(f"     - H recon loss:  {h_recon_weight:6.2f} ({h_recon_weight/total_weight*100:5.1f}%)")
        print(f"     - X recon loss:  {x_recon_weight:6.2f} ({x_recon_weight/total_weight*100:5.1f}%)")
        print(f"   Reference model created (frozen)")
        print(f"   Policy model trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M")
    
    def compute_dpo_loss(self, batch_w, batch_l):
        """
        Compute DPO loss for preference pairs
        Args:
            batch_w: winning sample batch (higher permeability)
            batch_l: losing sample batch (lower permeability)
        Returns:
            loss: DPO loss
            loss_dict: dictionary with loss components
        """
        if not hasattr(self, 'ref_model'):
            raise RuntimeError("DPO not setup! Call model.setup_dpo(beta) first")
        
        # Prepare forward arguments (atom_gt is optional for Prompt models)
        forward_args_w = {
            'X': batch_w['X'], 
            'S': batch_w['S'],
            'prompt': batch_w.get('prompt', None),
            'mask': batch_w['mask'],
            'position_ids': batch_w['position_ids'], 
            'lengths': batch_w['lengths'], 
            'atom_mask': batch_w['atom_mask'],
            'key_mask': batch_w.get('key_mask', None),
            'L': batch_w.get('L', None)
        }
        if 'atom_gt' in batch_w:
            forward_args_w['atom_gt'] = batch_w['atom_gt']
            
        forward_args_l = {
            'X': batch_l['X'], 
            'S': batch_l['S'],
            'prompt': batch_l.get('prompt', None),
            'mask': batch_l['mask'],
            'position_ids': batch_l['position_ids'], 
            'lengths': batch_l['lengths'], 
            'atom_mask': batch_l['atom_mask'],
            'key_mask': batch_l.get('key_mask', None),
            'L': batch_l.get('L', None)
        }
        if 'atom_gt' in batch_l:
            forward_args_l['atom_gt'] = batch_l['atom_gt']
        
        # This ensures fair comparison at the same diffusion timestep
        batch_size = batch_w['lengths'].shape[0]
        t_shared = torch.randint(
            0, self.diffusion.num_steps, 
            (batch_size,), 
            dtype=torch.long, 
            device=batch_w['X'].device
        )
        
        # Forward pass on winning sample (with shared timestep)
        loss_w, loss_dict_w = self.forward(**forward_args_w, t=t_shared)
        
        # Forward pass on losing sample (with same timestep)
        loss_l, loss_dict_l = self.forward(**forward_args_l, t=t_shared)
        
        # Reference model forward (no grad, also with shared timestep)
        with torch.no_grad():
            loss_w_ref, loss_dict_w_ref = self.ref_model.forward(**forward_args_w, t=t_shared)
            loss_l_ref, loss_dict_l_ref = self.ref_model.forward(**forward_args_l, t=t_shared)
        
        # DPO loss computation
        # loss_diff = (loss_w - loss_w_ref) - (loss_l - loss_l_ref)
        # We want winning sample to have lower loss, losing sample to have higher loss
        # Keep gradients for policy model (loss_w, loss_l), ref model already no_grad
        loss_diff = (loss_w - loss_w_ref) - (loss_l - loss_l_ref)
        dpo_loss_raw = -torch.nn.functional.logsigmoid(-self.dpo_beta * loss_diff).mean()
        
        # ⭐ 新方案：为DPO、H recon、X recon分别设置权重
        # 获取各loss权重（默认值为1.0）
        dpo_weight = getattr(self, 'dpo_loss_weight', 1.0)
        h_weight = getattr(self, 'h_recon_weight', 1.0)
        x_weight = getattr(self, 'x_recon_weight', 1.0)
        
        # 分别提取H和X的重建loss（使用winning样本）
        if 'H' in loss_dict_w and 'X' in loss_dict_w:
            # 注意：loss_dict中的H和X是原始loss（未乘h_loss_weight）
            h_recon_raw = loss_dict_w['H']  # 原始序列loss
            x_recon_raw = loss_dict_w['X']  # 原始结构loss
            
            # 加权后的三个loss组件（在相同scale上）
            dpo_loss_weighted = dpo_weight * dpo_loss_raw
            h_recon_weighted = h_weight * h_recon_raw
            x_recon_weighted = x_weight * x_recon_raw
            
            # 总loss = 三个组件的加权和
            total_loss = dpo_loss_weighted + h_recon_weighted + x_recon_weighted
            
            # 用于logging的reconstruction loss（未加权，兼容旧代码）
            reconstruction_loss = loss_w
        else:
            # 如果没有分解的loss，回退到旧方案
            total_loss = dpo_loss_raw + loss_w
            reconstruction_loss = loss_w
            h_recon_raw = None
            x_recon_raw = None
            dpo_loss_weighted = dpo_loss_raw
            h_recon_weighted = None
            x_recon_weighted = None
        
        # Combine losses for detailed logging
        dpo_loss_dict = {
            'dpo_total': total_loss,
            'dpo_only': dpo_loss_raw,  # 原始DPO loss（未加权）
            'dpo_weighted': dpo_loss_weighted,  # 加权后的DPO loss
            'reconstruction': reconstruction_loss,  # 兼容旧代码
            'dpo_weight': dpo_weight,
            'h_recon_weight': h_weight,
            'x_recon_weight': x_weight,
            'loss_w': loss_w,
            'loss_l': loss_l,
            'loss_w_ref': loss_w_ref,
            'loss_l_ref': loss_l_ref,
        }
        
        # 添加分解的recon loss
        if h_recon_raw is not None:
            dpo_loss_dict['h_recon_raw'] = h_recon_raw
            dpo_loss_dict['h_recon_weighted'] = h_recon_weighted
        if x_recon_raw is not None:
            dpo_loss_dict['x_recon_raw'] = x_recon_raw
            dpo_loss_dict['x_recon_weighted'] = x_recon_weighted
        
        # Add component-wise DPO losses if available (keep gradients)
        if 'H' in loss_dict_w:
            loss_diff_H = (loss_dict_w['H'] - loss_dict_w_ref['H']) - \
                         (loss_dict_l['H'] - loss_dict_l_ref['H'])
            dpo_loss_H = -torch.nn.functional.logsigmoid(-self.dpo_beta * loss_diff_H).mean()
            dpo_loss_dict['dpo_H'] = dpo_loss_H
            dpo_loss_dict['loss_w_H'] = loss_dict_w['H']
            dpo_loss_dict['loss_l_H'] = loss_dict_l['H']
        
        if 'X' in loss_dict_w:
            loss_diff_X = (loss_dict_w['X'] - loss_dict_w_ref['X']) - \
                         (loss_dict_l['X'] - loss_dict_l_ref['X'])
            dpo_loss_X = -torch.nn.functional.logsigmoid(-self.dpo_beta * loss_diff_X).mean()
            dpo_loss_dict['dpo_X'] = dpo_loss_X
            dpo_loss_dict['loss_w_X'] = loss_dict_w['X']
            dpo_loss_dict['loss_l_X'] = loss_dict_l['X']
        
        return total_loss, dpo_loss_dict