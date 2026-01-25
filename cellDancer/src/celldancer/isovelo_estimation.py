"""
IsoVelo: Isoform-level RNA velocity estimation module.

This module extends cellDancer to support isoform-level RNA velocity estimation,
where for each gene with K isoforms:
    du/dt = α(t) - Σ(k=1 to K) β_k(t) × u
    ds_k/dt = β_k(t) × u - γ_k(t) × s_k

Parameters estimated: α (gene-specific), β_k, γ_k (isoform-specific)
"""

import os
import sys
import glob
import shutil
import datetime
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from tqdm import tqdm
import pkg_resources
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
import logging

handle = 'IsoVelo'
logger_isovelo = logging.getLogger(handle)
logging.getLogger(handle).setLevel(logging.INFO)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from .sampling import downsampling_embedding, downsampling


class IsoVelo_DNN_layer(nn.Module):
    """
    Dynamic neural network for isoform-level velocity estimation.
    
    For a gene with K isoforms:
    - Input: (unsplice, splice_1, splice_2, ..., splice_K) -> K+1 dimensions
    - Output: (alpha, beta_1, ..., beta_K, gamma_1, ..., gamma_K) -> 2K+1 dimensions
    """

    def __init__(self, h1, h2, n_isoforms):
        super().__init__()
        self.n_isoforms = n_isoforms
        # Input: 1 (unsplice) + K (splice per isoform) = K+1
        self.l1 = nn.Linear(n_isoforms + 1, h1)
        self.l2 = nn.Linear(h1, h2)
        # Output: 1 (alpha) + K (beta) + K (gamma) = 2K+1
        self.l3 = nn.Linear(h2, 2 * n_isoforms + 1)

    def forward(self, unsplice, splices, alpha0, beta0s, gamma0s, dt):
        """
        Forward pass for isoform-level velocity estimation.
        
        Args:
            unsplice: Tensor of shape (n_cells,) - shared unspliced counts
            splices: Tensor of shape (n_cells, n_isoforms) - spliced counts per isoform
            alpha0: Scalar - initial alpha scaling factor
            beta0s: Tensor of shape (n_isoforms,) - initial beta scaling factors
            gamma0s: Tensor of shape (n_isoforms,) - initial gamma scaling factors
            dt: Time step
            
        Returns:
            unsplice_predict: Predicted unspliced counts
            splice_predicts: Predicted spliced counts per isoform
            alpha: Estimated alpha (gene-specific)
            betas: Estimated betas (isoform-specific)
            gammas: Estimated gammas (isoform-specific)
        """
        # Create input tensor: [unsplice, splice_1, ..., splice_K]
        # unsplice: (n_cells,), splices: (n_cells, n_isoforms)
        # Ensure proper tensor types
        if not isinstance(unsplice, torch.Tensor):
            unsplice = torch.tensor(unsplice, dtype=torch.float32)
        if not isinstance(splices, torch.Tensor):
            splices = torch.tensor(splices, dtype=torch.float32)
        
        input_tensor = torch.cat([unsplice.unsqueeze(1), splices], dim=1).float()
                
        # Forward through network
        x = self.l1(input_tensor)
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(x)
        x = self.l3(x)
        output = torch.sigmoid(x)
        
        # #region agent log H9 - forward output check
        has_nan_output = bool(torch.isnan(output).any())
        
        # Parse output: [alpha, beta_1, ..., beta_K, gamma_1, ..., gamma_K]
        alpha = output[:, 0]
        betas = output[:, 1:self.n_isoforms + 1]
        gammas = output[:, self.n_isoforms + 1:]
        
        # Scale by initial values
        alpha = alpha * alpha0
        betas = betas * beta0s.unsqueeze(0)  # broadcast across cells
        gammas = gammas * gamma0s.unsqueeze(0)
        
        # Compute predictions using the isoform model:
        # du/dt = α - Σ(β_k × u)
        # ds_k/dt = β_k × u - γ_k × s_k
        
        # Sum of all beta * u for unsplice dynamics
        sum_beta_u = torch.sum(betas * unsplice.unsqueeze(1), dim=1)
        unsplice_predict = unsplice + (alpha - sum_beta_u) * dt
        
        # Each isoform's splice dynamics
        splice_predicts = splices + (betas * unsplice.unsqueeze(1) - gammas * splices) * dt
        
        return unsplice_predict, splice_predicts, alpha, betas, gammas

    def save(self, model_path):
        torch.save({
            "l1": self.l1,
            "l2": self.l2,
            "l3": self.l3,
            "n_isoforms": self.n_isoforms
        }, model_path)

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.l1 = checkpoint["l1"]
        self.l2 = checkpoint["l2"]
        self.l3 = checkpoint["l3"]
        self.n_isoforms = checkpoint["n_isoforms"]


class IsoVelo_DNN_module(nn.Module):
    """
    Module for calculating loss function and predictions for isoform-level velocity.
    """
    
    def __init__(self, module, n_neighbors=None):
        super().__init__()
        self.module = module
        self.n_neighbors = n_neighbors

    def velocity_calculate(self,
                           unsplice,
                           splices,
                           alpha0,
                           beta0s,
                           gamma0s,
                           dt,
                           embedding1,
                           embedding2,
                           isoform_names=None,
                           barcode=None,
                           loss_func=None,
                           cost2_cutoff=None,
                           trace_cost_ratio=None,
                           corrcoef_cost_ratio=None):
        """
        Calculate velocity and loss for isoform-level model.
        
        Args:
            unsplice: Array of shape (n_cells,)
            splices: Array of shape (n_cells, n_isoforms)
            alpha0: Initial alpha value
            beta0s: Array of shape (n_isoforms,)
            gamma0s: Array of shape (n_isoforms,)
            dt: Time step
            embedding1, embedding2: Embedding coordinates
            isoform_names: List of isoform names
            loss_func: Loss function type
        """
        # Generate neighbor indices using embedding
        points = np.array([embedding1.numpy(), embedding2.numpy()]).transpose()
        
        self.n_neighbors = min((points.shape[0] - 1), self.n_neighbors)
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Convert to tensors
        unsplice = torch.tensor(unsplice) if not isinstance(unsplice, torch.Tensor) else unsplice
        splices = torch.tensor(splices) if not isinstance(splices, torch.Tensor) else splices
        indices = torch.tensor(indices)
        
        # Forward pass
        unsplice_predict, splice_predicts, alpha, betas, gammas = self.module(
            unsplice, splices, alpha0, beta0s, gamma0s, dt
        )
        
        n_isoforms = splices.shape[1]
        
        def cosine_similarity_isovelo(unsplice, splices, unsplice_predict, splice_predicts, indices):
            """
            Cosine similarity loss for isoform-level velocity.
            
            The velocity vector is in (K+1)-dimensional space: (du, ds_1, ..., ds_K)
            """
            # Velocity from current state to predicted state
            uv = unsplice_predict - unsplice  # (n_cells,)
            sv = splice_predicts - splices    # (n_cells, n_isoforms)
            
            # Velocity vector: concatenate (du, ds_1, ..., ds_K)
            v = torch.cat([uv.unsqueeze(1), sv], dim=1)  # (n_cells, K+1)
            
            # Neighbor displacement vectors
            # For each cell, compute displacement to its neighbors
            n_cells = unsplice.shape[0]
            n_neighbors = indices.shape[1] - 1  # exclude self
            
            cosine_values = []
            for i in range(n_cells):
                neighbor_idx = indices[i, 1:]  # exclude self (first index)
                
                # Displacement to neighbors
                un = unsplice[neighbor_idx] - unsplice[i]  # (n_neighbors,)
                sn = splices[neighbor_idx] - splices[i]    # (n_neighbors, n_isoforms)
                
                # Neighbor displacement vectors
                vn = torch.cat([un.unsqueeze(1), sn], dim=1)  # (n_neighbors, K+1)
                
                # Current cell's velocity
                vi = v[i]  # (K+1,)
                
                # Cosine similarity between vi and each neighbor displacement
                vi_norm = torch.norm(vi)
                vn_norm = torch.norm(vn, dim=1)
                
                # Avoid division by zero
                denom = vi_norm * vn_norm
                denom = torch.where(denom == 0, torch.ones_like(denom), denom)
                
                cos_sim = torch.sum(vi * vn, dim=1) / denom
                max_cos = torch.max(cos_sim)
                cosine_values.append(max_cos)
            
            cosine_tensor = torch.stack(cosine_values)
            return 1 - cosine_tensor
        
        def rmse_isovelo(unsplice, splices, unsplice_predict, splice_predicts, indices):
            """
            RMSE loss for isoform-level velocity.
            """
            # Velocity from current state to predicted state
            uv = unsplice_predict - unsplice
            sv = splice_predicts - splices
            v = torch.cat([uv.unsqueeze(1), sv], dim=1)
            
            n_cells = unsplice.shape[0]
            rmse_values = []
            
            for i in range(n_cells):
                neighbor_idx = indices[i, 1:]
                
                un = unsplice[neighbor_idx] - unsplice[i]
                sn = splices[neighbor_idx] - splices[i]
                vn = torch.cat([un.unsqueeze(1), sn], dim=1)
                
                vi = v[i]
                
                # RMSE between vi and each neighbor displacement
                diff = vi - vn
                rmse = torch.sqrt(torch.mean(diff ** 2, dim=1))
                min_rmse = torch.min(rmse)
                rmse_values.append(min_rmse)
            
            return torch.stack(rmse_values)
        
        # Calculate loss
        if loss_func == 'cosine':
            cost = cosine_similarity_isovelo(unsplice, splices, unsplice_predict, splice_predicts, indices)
            cost_fin = torch.mean(cost)
        elif loss_func == 'rmse':
            cost = rmse_isovelo(unsplice, splices, unsplice_predict, splice_predicts, indices)
            cost_fin = torch.mean(cost)
        else:
            # Default to cosine
            cost = cosine_similarity_isovelo(unsplice, splices, unsplice_predict, splice_predicts, indices)
            cost_fin = torch.mean(cost)
        
        return cost_fin, unsplice_predict, splice_predicts, alpha, betas, gammas

    def summary_para_validation(self, cost_mean):
        loss_df = pd.DataFrame({'cost': cost_mean}, index=[0])
        return loss_df

    def summary_para(self, unsplice, splices, unsplice_predict, splice_predicts, 
                     alpha, betas, gammas, cost, isoform_names):
        """
        Create summary DataFrame for isoform-level results.
        
        Returns a DataFrame with one row per cell per isoform.
        """
        n_cells = unsplice.shape[0]
        n_isoforms = len(isoform_names)
                
        # Create DataFrame with isoform-level data
        rows = []
        for k, isoform in enumerate(isoform_names):
            for i in range(n_cells):
                rows.append({
                    'cellIndex': i,
                    'isoform_name': isoform,
                    'unsplice': unsplice[i].item() if hasattr(unsplice[i], 'item') else unsplice[i],
                    'splice': splices[i, k].item() if hasattr(splices[i, k], 'item') else splices[i, k],
                    'unsplice_predict': unsplice_predict[i].item() if hasattr(unsplice_predict[i], 'item') else unsplice_predict[i],
                    'splice_predict': splice_predicts[i, k].item() if hasattr(splice_predicts[i, k], 'item') else splice_predicts[i, k],
                    'alpha': alpha[i].item() if hasattr(alpha[i], 'item') else alpha[i],
                    'beta': betas[i, k].item() if hasattr(betas[i, k], 'item') else betas[i, k],
                    'gamma': gammas[i, k].item() if hasattr(gammas[i, k], 'item') else gammas[i, k],
                    'loss': cost.item() if hasattr(cost, 'item') else cost
                })
        
        isovelo_df = pd.DataFrame(rows)
        
        return isovelo_df


class IsoVelo_ltModule(pl.LightningModule):
    """
    PyTorch Lightning module for training isoform-level velocity network.
    """
    
    def __init__(self,
                 backbone=None,
                 n_isoforms=None,
                 initial_zoom=2,
                 initial_strech=1,
                 learning_rate=None,
                 dt=None,
                 loss_func=None,
                 cost2_cutoff=0,
                 optimizer='Adam',
                 trace_cost_ratio=0,
                 corrcoef_cost_ratio=0,
                 cost_type='smooth',
                 average_cost_window_size=10,
                 smooth_weight=0.9):
        super().__init__()
        self.backbone = backbone
        self.n_isoforms = n_isoforms
        self.validation_loss_df = pd.DataFrame()
        self.test_isovelo_df = None
        self.test_loss_df = None
        self.initial_zoom = initial_zoom
        self.initial_strech = initial_strech
        self.learning_rate = learning_rate
        self.dt = dt
        self.loss_func = loss_func
        self.cost2_cutoff = cost2_cutoff
        self.optimizer_name = optimizer
        self.trace_cost_ratio = trace_cost_ratio
        self.corrcoef_cost_ratio = corrcoef_cost_ratio
        self.save_hyperparameters(ignore=['backbone'])
        self.get_loss = 1000
        self.cost_type = cost_type
        self.average_cost_window_size = average_cost_window_size
        self.cost_window = []
        self.smooth_weight = smooth_weight

    def save(self, model_path):
        self.backbone.module.save(model_path)

    def load(self, model_path):
        self.backbone.module.load(model_path)

    def configure_optimizers(self):
        if self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate,
                betas=(0.9, 0.999), eps=10**(-8),
                weight_decay=0.004, amsgrad=False
            )
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=0.8
            )
        return optimizer

    def training_step(self, batch, batch_idx):
        """Training step for isoform-level model."""
        (unsplices, splices_list, gene_names, isoform_names_list,
         unsplicemaxs, splicemaxs_list, embedding1s, embedding2s) = batch
        
        unsplice = unsplices[0]
        splices = splices_list[0]  # (n_cells, n_isoforms)
        gene_name = gene_names[0]
        # Fix: DataLoader wraps each isoform name in a single-element tuple
        # Structure is: [('iso1',), ('iso2',), ...] - need to extract strings
        isoform_names = []
        for item in isoform_names_list:
            if isinstance(item, tuple) and len(item) == 1:
                isoform_names.append(item[0])
            elif isinstance(item, str):
                isoform_names.append(item)
            else:
                isoform_names.append(str(item))
        unsplicemax = unsplicemaxs[0]
        splicemaxs = splicemaxs_list[0]  # (n_isoforms,)
        embedding1 = embedding1s[0]
        embedding2 = embedding2s[0]
        
        n_isoforms = splices.shape[1]
        
        # Initial parameter values
        alpha0 = np.float32(unsplicemax * self.initial_zoom)
        beta0s = torch.ones(n_isoforms, dtype=torch.float32)
        # Guard against 0/0 when unsplicemax or splicemaxs are zero.
        eps = np.float32(1e-8)
        gamma0s = torch.tensor(
            [
                (unsplicemax / (splicemaxs[k] if splicemaxs[k] > 0 else eps)) * self.initial_strech
                for k in range(n_isoforms)
            ],
            dtype=torch.float32
        )
        
        cost, unsplice_predict, splice_predicts, alpha, betas, gammas = \
            self.backbone.velocity_calculate(
                unsplice, splices, alpha0, beta0s, gamma0s, self.dt,
                embedding1, embedding2,
                isoform_names=isoform_names,
                loss_func=self.loss_func,
                cost2_cutoff=self.cost2_cutoff,
                trace_cost_ratio=self.trace_cost_ratio,
                corrcoef_cost_ratio=self.corrcoef_cost_ratio
            )
        
        # Smoothed loss tracking
        if self.cost_type == 'smooth':
            if self.get_loss == 1000:
                self.get_loss = cost
            smoothed_val = cost * self.smooth_weight + (1 - self.smooth_weight) * self.get_loss
            self.get_loss = smoothed_val
            self.log("loss", self.get_loss)
        else:
            self.get_loss = cost
            self.log("loss", self.get_loss)
        
        return {
            "loss": cost,
            "betas": betas.detach(),
            "gammas": gammas.detach()
        }

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        (unsplices, splices_list, gene_names, isoform_names_list,
         unsplicemaxs, splicemaxs_list, embedding1s, embedding2s) = batch
        
        gene_name = gene_names[0]
        
        if self.current_epoch != 0:
            cost = self.get_loss.data.numpy()
            loss_df = self.backbone.summary_para_validation(cost)
            loss_df.insert(0, "gene_name", gene_name)
            loss_df.insert(1, "epoch", self.current_epoch)
            if self.validation_loss_df.empty:
                self.validation_loss_df = loss_df
            else:
                self.validation_loss_df = pd.concat([self.validation_loss_df, loss_df], ignore_index=True)

    def test_step(self, batch, batch_idx):
        """Test step - generate predictions."""
        (unsplices, splices_list, gene_names, isoform_names_list,
         unsplicemaxs, splicemaxs_list, embedding1s, embedding2s) = batch
        
        unsplice = unsplices[0]
        splices = splices_list[0]
        gene_name = gene_names[0]
        
        # Fix: DataLoader wraps each isoform name in a single-element tuple
        # Structure is: [('iso1',), ('iso2',), ...] - need to extract strings
        isoform_names = []
        for item in isoform_names_list:
            if isinstance(item, tuple) and len(item) == 1:
                isoform_names.append(item[0])
            elif isinstance(item, str):
                isoform_names.append(item)
            else:
                isoform_names.append(str(item))
        
        unsplicemax = unsplicemaxs[0]
        splicemaxs = splicemaxs_list[0]
        embedding1 = embedding1s[0]
        embedding2 = embedding2s[0]
        
        n_isoforms = splices.shape[1]
        
        alpha0 = np.float32(unsplicemax * 2)
        beta0s = torch.ones(n_isoforms, dtype=torch.float32)
        # Guard against 0/0 when unsplicemax or splicemaxs are zero.
        eps = np.float32(1e-8)
        gamma0s = torch.tensor(
            [unsplicemax / (splicemaxs[k] if splicemaxs[k] > 0 else eps)
             for k in range(n_isoforms)],
            dtype=torch.float32
        )
        
        cost, unsplice_predict, splice_predicts, alpha, betas, gammas = \
            self.backbone.velocity_calculate(
                unsplice, splices, alpha0, beta0s, gamma0s, self.dt,
                embedding1, embedding2,
                isoform_names=isoform_names,
                loss_func=self.loss_func,
                cost2_cutoff=self.cost2_cutoff,
                trace_cost_ratio=self.trace_cost_ratio,
                corrcoef_cost_ratio=self.corrcoef_cost_ratio
            )
        
        self.test_isovelo_df = self.backbone.summary_para(
            unsplice, splices,
            unsplice_predict.data.numpy(), splice_predicts.data.numpy(),
            alpha.data.numpy(), betas.data.numpy(), gammas.data.numpy(),
            cost, isoform_names
        )
        
        self.test_isovelo_df.insert(1, "gene_name", gene_name)


class IsoVelo_getItem(Dataset):
    """
    Dataset class for isoform-level velocity estimation.
    
    Handles grouping by gene and organizing isoform data.
    """
    
    def __init__(self, data_fit=None, data_predict=None, datastatus="predict_dataset",
                 permutation_ratio=0.1, norm_u_s=True, norm_cell_distribution=False,
                 retain_nonzero=True):
        self.data_fit = data_fit
        self.data_predict = data_predict
        self.datastatus = datastatus
        self.permutation_ratio = permutation_ratio
        self.norm_u_s = norm_u_s
        self.norm_cell_distribution = norm_cell_distribution
        self.retain_nonzero = retain_nonzero
        
        # Get unique genes
        self.gene_names = list(data_fit.gene_name.drop_duplicates())
        
        # Build isoform mapping for each gene
        self.gene_isoform_map = {}
        for gene in self.gene_names:
            gene_data = data_fit[data_fit.gene_name == gene]
            isoforms = list(gene_data.isoform_name.drop_duplicates())
            self.gene_isoform_map[gene] = isoforms
        
        self.norm_max_unsplice = None
        self.norm_max_splices = None

    def __len__(self):
        return len(self.gene_names)

    def __getitem__(self, idx):
        gene_name = self.gene_names[idx]
        isoform_names = self.gene_isoform_map[gene_name]
        n_isoforms = len(isoform_names)
        
        if self.datastatus == "fit_dataset":
            gene_data = self.data_fit[self.data_fit.gene_name == gene_name]
            
            if self.permutation_ratio == 1:
                data = gene_data
            elif 0 < self.permutation_ratio < 1:
                # Sample cells (need to sample same cells across all isoforms)
                cellIDs = gene_data.cellID.drop_duplicates()
                sampled_cellIDs = cellIDs.sample(frac=self.permutation_ratio)
                if self.retain_nonzero:
                    nonzero_cellIDs = gene_data[
                        (gene_data.unsplice > 0) | (gene_data.splice > 0)
                    ].cellID.drop_duplicates()
                    sampled_cellIDs = pd.Index(sampled_cellIDs).union(nonzero_cellIDs)
                data = gene_data[gene_data.cellID.isin(sampled_cellIDs)]
            else:
                print('sampling ratio is wrong!')
                data = gene_data
        else:
            data = self.data_predict[self.data_predict.gene_name == gene_name]
        
        # Get prediction data for normalization
        data_pred = self.data_predict[self.data_predict.gene_name == gene_name]
        
        # Get unique cells (should be consistent across isoforms)
        cells = data.cellID.drop_duplicates().tolist()
        n_cells = len(cells)
        
        # Create cell index mapping
        cell_to_idx = {cell: i for i, cell in enumerate(cells)}
        
        # Prepare arrays
        # For unsplice: same value for all isoforms, so take from first isoform
        first_isoform = isoform_names[0]
        first_iso_data = data[data.isoform_name == first_isoform].set_index('cellID')
        first_iso_data = first_iso_data.loc[cells]  # Ensure consistent ordering
        
        unsplice = np.array(first_iso_data.unsplice.astype(np.float32))
        embedding1 = np.array(first_iso_data.embedding1.astype(np.float32))
        embedding2 = np.array(first_iso_data.embedding2.astype(np.float32))
        
        # For splices: different for each isoform
        splices = np.zeros((n_cells, n_isoforms), dtype=np.float32)
        for k, isoform in enumerate(isoform_names):
            iso_data = data[data.isoform_name == isoform].set_index('cellID')
            iso_data = iso_data.loc[cells]
            splices[:, k] = iso_data.splice.values.astype(np.float32)
        
        # Get max values for normalization from prediction data
        pred_first_iso = data_pred[data_pred.isoform_name == first_isoform]
        unsplicemax = np.float32(pred_first_iso.unsplice.max())
        
        splicemaxs = np.zeros(n_isoforms, dtype=np.float32)
        for k, isoform in enumerate(isoform_names):
            iso_pred = data_pred[data_pred.isoform_name == isoform]
            splicemaxs[k] = iso_pred.splice.max()
        
        # Normalize if requested
        if self.norm_u_s:
            unsplice = unsplice / unsplicemax if unsplicemax > 0 else unsplice
            for k in range(n_isoforms):
                if splicemaxs[k] > 0:
                    splices[:, k] = splices[:, k] / splicemaxs[k]
        
        return (unsplice, splices, gene_name, isoform_names,
                unsplicemax, splicemaxs, embedding1, embedding2)


class IsoVelo_feedData(pl.LightningDataModule):
    """
    DataLoader module for isoform-level velocity estimation.
    """
    
    def __init__(self, data_fit=None, data_predict=None, permutation_ratio=1,
                 norm_u_s=True, norm_cell_distribution=False, retain_nonzero=True):
        super().__init__()
        
        self.fit_dataset = IsoVelo_getItem(
            data_fit=data_fit, data_predict=data_predict,
            datastatus="fit_dataset", permutation_ratio=permutation_ratio,
            norm_u_s=norm_u_s, norm_cell_distribution=norm_cell_distribution,
            retain_nonzero=retain_nonzero
        )
        
        self.predict_dataset = IsoVelo_getItem(
            data_fit=data_fit, data_predict=data_predict,
            datastatus="predict_dataset", permutation_ratio=permutation_ratio,
            norm_u_s=norm_u_s, retain_nonzero=retain_nonzero
        )

    def subset(self, indices):
        import copy
        temp = copy.copy(self)
        temp.fit_dataset = Subset(self.fit_dataset, indices)
        temp.predict_dataset = Subset(self.predict_dataset, indices)
        return temp

    def train_dataloader(self):
        return DataLoader(self.fit_dataset, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.fit_dataset, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.predict_dataset, num_workers=0)


def _isovelo_train_thread(datamodule,
                          data_indices,
                          save_path=None,
                          max_epoches=None,
                          check_val_every_n_epoch=None,
                          norm_u_s=None,
                          patience=None,
                          learning_rate=None,
                          dt=None,
                          loss_func=None,
                          n_neighbors=None,
                          model_save_path=None):
    """
    Training thread for a single gene's isoform-level velocity estimation.
    """
    try:
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        selected_data = datamodule.subset(data_indices)
        
        # Get data to determine number of isoforms
        (unsplice, splices, gene_name, isoform_names,
         unsplicemax, splicemaxs, embedding1, embedding2) = selected_data.fit_dataset.__getitem__(0)
        
        n_isoforms = len(isoform_names)
        
        # Initialize network with correct number of isoforms
        backbone = IsoVelo_DNN_module(
            IsoVelo_DNN_layer(100, 100, n_isoforms),
            n_neighbors=n_neighbors
        )
        model = IsoVelo_ltModule(
            backbone=backbone,
            n_isoforms=n_isoforms,
            dt=dt,
            learning_rate=learning_rate,
            loss_func=loss_func
        )
        
        early_stop_callback = EarlyStopping(
            monitor="loss", min_delta=0.0, patience=patience, mode='min'
        )
        
        if check_val_every_n_epoch is None:
            trainer = pl.Trainer(
                max_epochs=max_epoches,
                progress_bar_refresh_rate=0,
                reload_dataloaders_every_n_epochs=1,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
            )
        else:
            trainer = pl.Trainer(
                max_epochs=max_epoches,
                progress_bar_refresh_rate=0,
                reload_dataloaders_every_n_epochs=1,
                logger=False,
                enable_checkpointing=False,
                check_val_every_n_epoch=check_val_every_n_epoch,
                enable_model_summary=False,
                callbacks=[early_stop_callback]
            )
        
        if max_epoches > 0:
            trainer.fit(model, selected_data)
        
        trainer.test(model, selected_data, verbose=False)
        
        if model_save_path is not None:
            model.save(model_save_path)
        
        loss_df = model.validation_loss_df
        isovelo_df = model.test_isovelo_df
        
        # Denormalize if needed
        if norm_u_s:
            isovelo_df['unsplice'] = isovelo_df['unsplice'] * unsplicemax
            isovelo_df['unsplice_predict'] = isovelo_df['unsplice_predict'] * unsplicemax
            
            # Denormalize splice values per isoform
            for k, isoform in enumerate(isoform_names):
                mask = isovelo_df['isoform_name'] == isoform
                isovelo_df.loc[mask, 'splice'] = isovelo_df.loc[mask, 'splice'] * splicemaxs[k]
                isovelo_df.loc[mask, 'splice_predict'] = isovelo_df.loc[mask, 'splice_predict'] * splicemaxs[k]
                isovelo_df.loc[mask, 'beta'] = isovelo_df.loc[mask, 'beta'] * unsplicemax
                isovelo_df.loc[mask, 'gamma'] = isovelo_df.loc[mask, 'gamma'] * splicemaxs[k]
        
        if model_save_path is not None:
            model.save(model_save_path)
        
        # Save results
        header_loss_df = ['gene_name', 'epoch', 'loss']
        header_isovelo_df = ['cellIndex', 'gene_name', 'isoform_name', 'unsplice', 'splice',
                            'unsplice_predict', 'splice_predict', 'alpha', 'beta', 'gamma', 'loss']
        
        loss_df.to_csv(
            os.path.join(save_path, 'TEMP', f'loss_{gene_name}.csv'),
            header=header_loss_df, index=False
        )
        isovelo_df.to_csv(
            os.path.join(save_path, 'TEMP', f'isovelo_estimation_{gene_name}.csv'),
            header=header_isovelo_df, index=False
        )
        
        return None
        
    except Exception as e:
        print(f"Error processing gene {gene_name}: {e}")
        return gene_name


def build_isovelo_datamodule(isovelo_u_s,
                             speed_up,
                             norm_u_s,
                             permutation_ratio,
                             norm_cell_distribution=False,
                             gene_list=None,
                             downsample_method='neighbors',
                             n_neighbors_downsample=30,
                             step=(200, 200),
                             downsample_target_amount=None,
                             retain_nonzero=True):
    """
    Build data module for isoform-level velocity estimation.
    """
    step_i = step[0]
    step_j = step[1]
    
    required_cols = ['gene_name', 'isoform_name', 'unsplice', 'splice',
                     'embedding1', 'embedding2', 'cellID']
    
    if gene_list is None:
        data_df = isovelo_u_s[required_cols]
    else:
        data_df = isovelo_u_s[required_cols][isovelo_u_s.gene_name.isin(gene_list)]
    
    if speed_up:
        # Downsample based on first gene's first isoform
        first_gene = list(data_df.gene_name.drop_duplicates())[0]
        first_gene_data = data_df[data_df.gene_name == first_gene]
        first_isoform = list(first_gene_data.isoform_name.drop_duplicates())[0]
        sample_data = first_gene_data[first_gene_data.isoform_name == first_isoform]
        
        # Create temporary df for downsampling
        sample_df = sample_data[['gene_name', 'unsplice', 'splice', 'embedding1', 'embedding2', 'cellID']].copy()
        
        _, sampling_ixs, _ = downsampling_embedding(
            sample_df,
            para=downsample_method,
            target_amount=downsample_target_amount,
            step=(step_i, step_j),
            n_neighbors=n_neighbors_downsample,
            projection_neighbor_choice='embedding'
        )
        
        downsample_cellid = sample_data.cellID.iloc[sampling_ixs]
        keep_cellid = pd.Index(downsample_cellid)
        if retain_nonzero:
            nonzero_cellid = data_df[
                (data_df.unsplice > 0) | (data_df.splice > 0)
            ].cellID.drop_duplicates()
            keep_cellid = keep_cellid.union(nonzero_cellid)
        data_df_downsampled = data_df[data_df.cellID.isin(keep_cellid)]
        
        feed_data = IsoVelo_feedData(
            data_fit=data_df_downsampled,
            data_predict=data_df,
            permutation_ratio=permutation_ratio,
            norm_u_s=norm_u_s,
            norm_cell_distribution=norm_cell_distribution,
            retain_nonzero=retain_nonzero
        )
    else:
        feed_data = IsoVelo_feedData(
            data_fit=data_df,
            data_predict=data_df,
            permutation_ratio=permutation_ratio,
            norm_u_s=norm_u_s,
            norm_cell_distribution=norm_cell_distribution,
            retain_nonzero=retain_nonzero
        )
    
    return feed_data


def isovelo_velocity(
    isovelo_u_s,
    gene_list=None,
    max_epoches=200,
    check_val_every_n_epoch=10,
    patience=3,
    learning_rate=0.001,
    dt=0.5,
    n_neighbors=30,
    permutation_ratio=0.125,
    speed_up=True,
    norm_u_s=True,
    norm_cell_distribution=True,
    loss_func='cosine',
    n_jobs=-1,
    save_path=None,
    retain_nonzero=True,
):
    """
    Isoform-level velocity estimation for each cell.
    
    Arguments
    ---------
    isovelo_u_s: `pandas.DataFrame`
        Dataframe containing isoform-level data with columns:
        ['gene_name', 'isoform_name', 'unsplice', 'splice', 'cellID', 'clusters', 'embedding1', 'embedding2']
        
        For each gene, unsplice should be the same across all isoforms for a given cell.
        
    gene_list: optional, `list` (default: None)
        Gene list for velocity estimation. `None` to estimate for all genes.
        
    max_epoches: optional, `int` (default: 200)
        Maximum number of training epochs.
        
    check_val_every_n_epoch: optional, `int` (default: 10)
        Check loss every n epochs.
        
    patience: optional, `int` (default: 3)
        Early stopping patience.
        
    dt: optional, `float` (default: 0.5)
        Time step size.
        
    permutation_ratio: optional, `float` (default: 0.125)
        Sampling ratio of cells in each epoch.
        
    speed_up: optional, `bool` (default: True)
        Whether to speed up by downsampling cells.
        
    norm_u_s: optional, `bool` (default: True)
        Whether to normalize unspliced and spliced reads.
        
    norm_cell_distribution: optional, `bool` (default: True)
        Whether to remove cell distribution bias.
        
    retain_nonzero: optional, `bool` (default: True)
        Ensure cells with nonzero unsplice/splice are always kept in sampling.
        
    loss_func: optional, `str` (default: 'cosine')
        Loss function type: 'cosine' or 'rmse'.
        
    n_jobs: optional, `int` (default: -1)
        Number of parallel jobs.
        
    save_path: optional, `str` (default: None)
        Path to save results.
        
    Returns
    -------
    loss_df: `pandas.DataFrame`
        Training loss records.
        
    isovelo_df: `pandas.DataFrame`
        Isoform-level velocity estimation results with columns:
        ['cellIndex', 'gene_name', 'isoform_name', 'unsplice', 'splice',
         'unsplice_predict', 'splice_predict', 'alpha', 'beta', 'gamma', 'loss',
         'cellID', 'clusters', 'embedding1', 'embedding2']
    """
    
    # Set output directory
    datestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    folder_name = 'isovelo_velocity_' + datestring
    
    if save_path is None:
        save_path = os.getcwd()
    
    try:
        shutil.rmtree(os.path.join(save_path, folder_name))
    except:
        os.mkdir(os.path.join(save_path, folder_name))
    save_path = os.path.join(save_path, folder_name)
    print('Using ' + save_path + ' as the output path.')
    
    try:
        shutil.rmtree(os.path.join(save_path, 'TEMP'))
    except:
        os.mkdir(os.path.join(save_path, 'TEMP'))
    
    # Set gene_list if not given
    if gene_list is None:
        gene_list = list(isovelo_u_s.gene_name.drop_duplicates())
    else:
        isovelo_u_s = isovelo_u_s[isovelo_u_s.gene_name.isin(gene_list)]
        all_gene_names = list(isovelo_u_s.gene_name.drop_duplicates())
        gene_not_in_data = list(set(gene_list).difference(set(all_gene_names)))
        gene_list = list(set(all_gene_names).intersection(set(gene_list)))
        if len(gene_not_in_data) > 0:
            print(gene_not_in_data, " not in the data isovelo_u_s")
    
    isovelo_u_s = isovelo_u_s.reset_index(drop=True)
    
    # Burning run with first gene
    gene_list_burning = [list(isovelo_u_s.gene_name.drop_duplicates())[0]]
    datamodule = build_isovelo_datamodule(
        isovelo_u_s, speed_up, norm_u_s, permutation_ratio,
        norm_cell_distribution, gene_list=gene_list_burning,
        retain_nonzero=retain_nonzero
    )
    
    result = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_isovelo_train_thread)(
            datamodule=datamodule,
            data_indices=[data_index],
            max_epoches=max_epoches,
            check_val_every_n_epoch=check_val_every_n_epoch,
            patience=patience,
            learning_rate=learning_rate,
            n_neighbors=n_neighbors,
            dt=dt,
            loss_func=loss_func,
            save_path=save_path,
            norm_u_s=norm_u_s
        )
        for data_index in range(0, len(gene_list_burning))
    )
    
    # Clean directory
    shutil.rmtree(os.path.join(save_path, 'TEMP'))
    os.mkdir(os.path.join(save_path, 'TEMP'))
    
    data_len = len(gene_list)
    
    # Create batches for parallel processing
    id_ranges = []
    if n_jobs == -1:
        interval = os.cpu_count()
    else:
        interval = n_jobs
    for i in range(0, data_len, interval):
        idx_start = i
        idx_end = min(i + interval, data_len)
        id_ranges.append((idx_start, idx_end))
    
    print('Arranging genes for parallel job.')
    if len(id_ranges) == 1:
        print(f'{data_len} gene(s) arranged to {len(id_ranges)} portion.')
    else:
        print(f'{data_len} genes arranged to {len(id_ranges)} portions.')
    
    unpredicted_gene_lst = []
    for id_range in tqdm(id_ranges, desc="IsoVelo Estimation", total=len(id_ranges),
                         position=1, leave=False, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        gene_list_batch = gene_list[id_range[0]:id_range[1]]
        datamodule = build_isovelo_datamodule(
            isovelo_u_s, speed_up, norm_u_s, permutation_ratio,
            norm_cell_distribution, gene_list=gene_list_batch,
            retain_nonzero=retain_nonzero
        )
        
        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_isovelo_train_thread)(
                datamodule=datamodule,
                data_indices=[data_index],
                max_epoches=max_epoches,
                check_val_every_n_epoch=check_val_every_n_epoch,
                n_neighbors=n_neighbors,
                dt=dt,
                loss_func=loss_func,
                learning_rate=learning_rate,
                patience=patience,
                save_path=save_path,
                norm_u_s=norm_u_s
            )
            for data_index in range(0, len(gene_list_batch))
        )
        
        # Collect unpredicted genes
        gene_name_lst = [x for x in result if x is not None]
        for gene in gene_name_lst:
            unpredicted_gene_lst.append(gene)
    
    if len(unpredicted_gene_lst) != 0:
        not_pred_err = 'Not predicted gene list: ' + str(unpredicted_gene_lst) + \
                       '. Try visualizing the unspliced and spliced columns to check quality.'
        logger_isovelo.error(not_pred_err)
    
    # Summarize results
    isovelo_df_pattern = os.path.join(save_path, 'TEMP', "isovelo_estimation*.csv")
    isovelo_df_files = glob.glob(isovelo_df_pattern)
    loss_df_pattern = os.path.join(save_path, 'TEMP', "loss*.csv")
    loss_df_files = glob.glob(loss_df_pattern)
    
    def combine_csv(save_file_path, files):
        with open(save_file_path, "wb") as fout:
            with open(files[0], "rb") as f:
                fout.write(f.read())
            for filepath in files[1:]:
                with open(filepath, "rb") as f:
                    next(f)
                    fout.write(f.read())
        return pd.read_csv(save_file_path)
    
    if len(isovelo_df_files) == 0:
        logger_isovelo.error('None of the genes were predicted. Check data quality.')
        return None, None
    else:
        isovelo_df = combine_csv(
            os.path.join(save_path, "isovelo_estimation.csv"),
            isovelo_df_files
        )
        loss_df = combine_csv(
            os.path.join(save_path, "loss.csv"),
            loss_df_files
        )
        
        shutil.rmtree(os.path.join(save_path, 'TEMP'))
        
        isovelo_df = isovelo_df.sort_values(
            by=['gene_name', 'isoform_name', 'cellIndex'],
            ascending=[True, True, True]
        )
        
        # Add cell metadata
        first_gene = isovelo_u_s.gene_name.iloc[0]
        first_isoform = isovelo_u_s[isovelo_u_s.gene_name == first_gene].isoform_name.iloc[0]
        onegene_oneiso = isovelo_u_s[
            (isovelo_u_s.gene_name == first_gene) & 
            (isovelo_u_s.isoform_name == first_isoform)
        ]
        embedding_info = onegene_oneiso[['cellID', 'clusters', 'embedding1', 'embedding2']]
        
        # Calculate number of isoforms for each gene
        gene_isoform_counts = isovelo_u_s.groupby('gene_name')['isoform_name'].nunique()
        
        # Repeat embedding info for each gene-isoform combination
        embedding_rows = []
        for gene in isovelo_df.gene_name.drop_duplicates():
            n_isoforms = gene_isoform_counts[gene]
            gene_embedding = pd.concat([embedding_info] * n_isoforms, ignore_index=True)
            embedding_rows.append(gene_embedding)
        
        embedding_col = pd.concat(embedding_rows, ignore_index=True)
        embedding_col.index = isovelo_df.index
        isovelo_df = pd.concat([isovelo_df, embedding_col], axis=1)
        
        isovelo_df.to_csv(
            os.path.join(save_path, 'isovelo_estimation.csv'),
            index=False
        )
        loss_df.to_csv(
            os.path.join(save_path, 'loss.csv'),
            index=False
        )
        
        return loss_df, isovelo_df
