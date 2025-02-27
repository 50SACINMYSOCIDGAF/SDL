import os
from collections import defaultdict
from typing import Dict
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.nn.utils.parametrizations import orthogonal
from torch.linalg import svd, qr
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

@dataclass
class FeatureMetrics:
    sparsity: float
    coherence: float
    stability: float
    diversity: float
    semantic_consistency: float
    recon_error: float
    ortho_loss: float

class AdaptiveSparsityController:
    def __init__(self, target_sparsity: float = 0.2, adaptation_rate: float = 0.05):
        self.target_sparsity = target_sparsity
        self.adaptation_rate = adaptation_rate
        self.moving_avg = None
        self.beta = 0.9
        self.min_penalty = 0.01
        self.max_penalty = 5.0
        self.history = []

    def compute_penalty(self, current_sparsity: float) -> torch.Tensor:
        if self.moving_avg is None:
            self.moving_avg = current_sparsity
        else:
            self.moving_avg = self.beta * self.moving_avg + (1 - self.beta) * current_sparsity
        error = self.target_sparsity - self.moving_avg
        self.history.append(error)
        penalty = torch.tanh(torch.tensor(error * 5))
        return torch.clamp(self.min_penalty + (self.max_penalty - self.min_penalty) * (penalty + 1) / 2, self.min_penalty, self.max_penalty)

    def update_target(self, epoch: int, total_epochs: int) -> None:
        progress = epoch / total_epochs
        base_sparsity = 0.15
        max_sparsity = 0.35
        cos_val = 0.5 * (1 + np.cos(progress * np.pi))
        self.target_sparsity = base_sparsity + (max_sparsity - base_sparsity) * (1 - cos_val)

class StiefelGrassmannianDictionary(nn.Module):
    def __init__(self, input_dim: int, dict_size: int, eps: float = 1e-6, tau: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.tau = tau
        self.eps = eps
        self.dictionary = orthogonal(nn.Linear(input_dim, dict_size, bias=False))
        self.initialize_stiefel()

    def initialize_stiefel(self) -> None:
        with torch.no_grad():
            if self.input_dim >= self.dict_size:
                rand_mat = torch.randn(self.input_dim, self.dict_size, device=self.dictionary.weight.device)
                Q, _ = qr(rand_mat)
                self.dictionary.weight.data = Q.t()
            else:
                rand_mat = torch.randn(self.dict_size, self.input_dim, device=self.dictionary.weight.device)
                Q, _ = qr(rand_mat)
                self.dictionary.weight.data = Q[:self.input_dim].t()

    def project_to_stiefel(self) -> None:
        with torch.no_grad():
            W = self.dictionary.weight
            U, _, Vh = svd(W.t(), full_matrices=False)
            self.dictionary.weight.data = Vh.t() @ U.t()

    def compute_frame_potential(self) -> torch.Tensor:
        W = self.dictionary.weight
        if self.input_dim >= self.dict_size:
            gram = W @ W.t()
            identity = torch.eye(gram.size(0), device=gram.device)
        else:
            gram = W.t() @ W
            identity = torch.eye(gram.size(0), device=gram.device)
        return torch.norm(gram - identity) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dictionary(x)

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, coherence_penalty: float = 0.1, diversity_weight: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.encoder = StiefelGrassmannianDictionary(input_dim, hidden_dim, eps=self.eps)
        self.decoder = StiefelGrassmannianDictionary(hidden_dim, input_dim, eps=self.eps)
        self.activation = nn.ReLU()
        self.coherence_penalty = coherence_penalty
        self.diversity_weight = diversity_weight
        self.sparsity_controller = AdaptiveSparsityController()
        self.semantic_cache = {}
        self.feature_ema = None
        self.ema_decay = 0.99
        self.buffer_size = 10000
        self.training_buffer = None
        self.buffer_filled = False

    def update_training_buffer(self, x: torch.Tensor):
        if not self.buffer_filled:
            x_float = x.detach().to(torch.float32)
            if self.training_buffer is None:
                self.training_buffer = x_float.clone()
            else:
                self.training_buffer = torch.cat([self.training_buffer, x_float], dim=0)
                if self.training_buffer.size(0) >= self.buffer_size:
                    self.buffer_filled = True
                    self.initialize_pca()

    def reinitialize_unstable_features(self, activations: torch.Tensor, stability_threshold: float = 0.2):
        with torch.no_grad():
            act_norm = activations / (torch.norm(activations, dim=1, keepdim=True) + self.eps)
            feature_corr = torch.mm(act_norm.T, act_norm)
            corr_mask = (feature_corr > 0.9) & (feature_corr < 1.0)
            for i in range(activations.size(1)):
                if self.feature_ema is not None:
                    feat_ema = self.feature_ema[i]
                    feat_current = activations[:, i].mean()
                    relative_change = abs(feat_current - feat_ema) / (abs(feat_ema) + self.eps)
                    stability = 1.0 / (1.0 + relative_change)
                    has_correlations = corr_mask[i].any()
                    if stability < stability_threshold or has_correlations:
                        rand_vec = torch.randn(self.encoder.input_dim, device=activations.device)
                        for j in range(i):
                            if j != i:
                                proj = torch.dot(rand_vec, self.encoder.dictionary.weight.data[j])
                                rand_vec -= proj * self.encoder.dictionary.weight.data[j]
                        rand_vec = rand_vec / torch.norm(rand_vec)
                        self.encoder.dictionary.weight.data[i] = rand_vec
                        if i < self.decoder.input_dim:
                            self.decoder.dictionary.weight.data[:, i] = rand_vec

    def compute_feature_statistics(self, activations: torch.Tensor):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with autocast(device_type, enabled=False):
            act = activations.float()
            batch_size = act.size(0)
            act_norm = act / (torch.norm(act, dim=1, keepdim=True) + self.eps)
            gram = torch.mm(act_norm, act_norm.t())
            mask = torch.triu(torch.ones_like(gram), diagonal=1).bool()
            coherence_mean = torch.abs(gram[mask]).mean()
            cov_matrix = (act.T @ act) / (batch_size - 1)
            eye = torch.eye(cov_matrix.size(0), device=cov_matrix.device)
            eigenvals = torch.linalg.eigvalsh(cov_matrix + self.eps * eye)
            normalized_eigenvals = eigenvals / (torch.sum(eigenvals) + self.eps)
            diversity = 1.0 - torch.max(normalized_eigenvals) / (torch.sum(normalized_eigenvals) + self.eps)
            singular_values = torch.linalg.svdvals(act).float()
            effective_rank = (singular_values / singular_values[0]).sum()
            return coherence_mean.item(), diversity.item(), (effective_rank / act.size(1)).item()

    def track_semantic_consistency(self, activations: torch.Tensor, batch_id: str):
        act_norm = activations / (torch.norm(activations, dim=1, keepdim=True) + self.eps)
        if batch_id in self.semantic_cache:
            prev_norm = self.semantic_cache[batch_id]
            consistency = torch.mean(torch.sum(act_norm * prev_norm, dim=1))
            self.semantic_cache[batch_id] = 0.9 * prev_norm + 0.1 * act_norm.detach()
            return consistency.item()
        self.semantic_cache[batch_id] = act_norm.detach()
        return 1.0

    def initialize_pca(self):
        with torch.no_grad():
            device = self.training_buffer.device
            buffer = self.training_buffer.cpu().float()
            centered = buffer - buffer.mean(dim=0, keepdim=True)
            cov = centered.T @ centered / (centered.size(0) - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvectors = eigenvectors[:, sorted_indices]
            components = eigenvectors[:, :self.encoder.dict_size]
            orig_dtype = self.encoder.dictionary.weight.dtype
            self.encoder.dictionary.weight.data = components.T.to(device=device, dtype=orig_dtype)
            self.decoder.dictionary.weight.data = components.to(device=device, dtype=orig_dtype)
            self.training_buffer = None

    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)
        if self.training and not self.buffer_filled:
            self.update_training_buffer(x)
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_scaled = x_centered / (torch.norm(x_centered, dim=1, keepdim=True) + self.eps)
        encoded = self.encoder(x_scaled)
        abs_encoded = torch.abs(encoded).float()
        if self.training:
            feature_means = abs_encoded.mean(dim=0)
            threshold = torch.quantile(abs_encoded, 0.8, dim=1, keepdim=True).detach()
            threshold = torch.maximum(threshold, 0.1 * feature_means.view(1, -1))
            soft_temp = torch.exp(-5 * feature_means).view(1, -1) * 0.1
            soft_threshold = torch.sigmoid((abs_encoded - threshold) / soft_temp)
            activations = encoded * soft_threshold
            noise_mask = activations.abs().mean(dim=1) < self.eps
            if noise_mask.any():
                noise_scale = 0.01 * threshold.mean()
                noise = torch.randn_like(activations[noise_mask]) * noise_scale
                activations[noise_mask] += noise
            self.reinitialize_unstable_features(activations)
        else:
            activations = encoded * (abs_encoded > torch.quantile(abs_encoded, 0.8, dim=1, keepdim=True)).float()
        decoded = self.decoder(activations)
        output = decoded * torch.norm(x_centered, dim=1, keepdim=True) + x.mean(dim=1, keepdim=True)
        return output, activations

class GatedSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, coherence_penalty: float = 0.1,
                 diversity_weight: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.encoder = StiefelGrassmannianDictionary(input_dim, hidden_dim, eps=self.eps)
        self.decoder = StiefelGrassmannianDictionary(hidden_dim, input_dim, eps=self.eps)

        # Context detection and gating networks
        self.context_projector = nn.Linear(input_dim, hidden_dim // 2)
        self.gate_generator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )

        self.activation = nn.ReLU()
        self.coherence_penalty = coherence_penalty
        self.diversity_weight = diversity_weight
        self.sparsity_controller = AdaptiveSparsityController()
        self.semantic_cache = {}
        self.feature_ema = None
        self.ema_decay = 0.99
        self.buffer_size = 10000
        self.training_buffer = None
        self.buffer_filled = False
        self.gate_strength = nn.Parameter(torch.ones(1) * 0.5)  # Learnable gate strength

    def update_training_buffer(self, x: torch.Tensor):
        if not self.buffer_filled:
            x_float = x.detach().to(torch.float32)
            if self.training_buffer is None:
                self.training_buffer = x_float.clone()
            else:
                self.training_buffer = torch.cat([self.training_buffer, x_float], dim=0)
                if self.training_buffer.size(0) >= self.buffer_size:
                    self.buffer_filled = True
                    self.initialize_pca()

    def reinitialize_unstable_features(self, activations: torch.Tensor, stability_threshold: float = 0.2):
        with torch.no_grad():
            act_norm = activations / (torch.norm(activations, dim=1, keepdim=True) + self.eps)
            feature_corr = torch.mm(act_norm.T, act_norm)
            corr_mask = (feature_corr > 0.9) & (feature_corr < 1.0)
            for i in range(activations.size(1)):
                if self.feature_ema is not None:
                    feat_ema = self.feature_ema[i]
                    feat_current = activations[:, i].mean()
                    relative_change = abs(feat_current - feat_ema) / (abs(feat_ema) + self.eps)
                    stability = 1.0 / (1.0 + relative_change)
                    has_correlations = corr_mask[i].any()
                    if stability < stability_threshold or has_correlations:
                        rand_vec = torch.randn(self.encoder.input_dim, device=activations.device)
                        for j in range(i):
                            if j != i:
                                proj = torch.dot(rand_vec, self.encoder.dictionary.weight.data[j])
                                rand_vec -= proj * self.encoder.dictionary.weight.data[j]
                        rand_vec = rand_vec / torch.norm(rand_vec)
                        self.encoder.dictionary.weight.data[i] = rand_vec
                        if i < self.decoder.input_dim:
                            self.decoder.dictionary.weight.data[:, i] = rand_vec

    def compute_feature_statistics(self, activations: torch.Tensor):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with autocast(device_type, enabled=False):
            act = activations.float()
            batch_size = act.size(0)
            act_norm = act / (torch.norm(act, dim=1, keepdim=True) + self.eps)
            gram = torch.mm(act_norm, act_norm.t())
            mask = torch.triu(torch.ones_like(gram), diagonal=1).bool()
            coherence_mean = torch.abs(gram[mask]).mean()
            cov_matrix = (act.T @ act) / (batch_size - 1)
            eye = torch.eye(cov_matrix.size(0), device=cov_matrix.device)
            eigenvals = torch.linalg.eigvalsh(cov_matrix + self.eps * eye)
            normalized_eigenvals = eigenvals / (torch.sum(eigenvals) + self.eps)
            diversity = 1.0 - torch.max(normalized_eigenvals) / (torch.sum(normalized_eigenvals) + self.eps)
            singular_values = torch.linalg.svdvals(act).float()
            effective_rank = (singular_values / singular_values[0]).sum()
            return coherence_mean.item(), diversity.item(), (effective_rank / act.size(1)).item()

    def track_semantic_consistency(self, activations: torch.Tensor, batch_id: str):
        act_norm = activations / (torch.norm(activations, dim=1, keepdim=True) + self.eps)
        if batch_id in self.semantic_cache:
            prev_norm = self.semantic_cache[batch_id]
            consistency = torch.mean(torch.sum(act_norm * prev_norm, dim=1))
            self.semantic_cache[batch_id] = 0.9 * prev_norm + 0.1 * act_norm.detach()
            return consistency.item()
        self.semantic_cache[batch_id] = act_norm.detach()
        return 1.0

    def initialize_pca(self):
        with torch.no_grad():
            device = self.training_buffer.device
            buffer = self.training_buffer.cpu().float()
            centered = buffer - buffer.mean(dim=0, keepdim=True)
            cov = centered.T @ centered / (centered.size(0) - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvectors = eigenvectors[:, sorted_indices]
            components = eigenvectors[:, :self.encoder.dict_size]
            orig_dtype = self.encoder.dictionary.weight.dtype
            self.encoder.dictionary.weight.data = components.T.to(device=device, dtype=orig_dtype)
            self.decoder.dictionary.weight.data = components.to(device=device, dtype=orig_dtype)
            self.training_buffer = None

    def compute_nonlinearity_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a score for potential semantic nonlinearity in the input."""
        # Project to lower dimension for context detection
        context_feats = self.context_projector(x)
        # Detect patterns indicative of negation or semantic contradiction
        context_norm = context_feats / (torch.norm(context_feats, dim=1, keepdim=True) + self.eps)
        # Compute self-attention between features to detect inconsistent patterns
        sim_matrix = torch.mm(context_norm, context_norm.t())
        # High variance in similarity indicates potential nonlinear semantics
        row_variance = torch.var(sim_matrix, dim=1)
        # Normalize and shape as a per-instance nonlinearity score
        nonlinearity_score = torch.sigmoid(row_variance - torch.mean(row_variance))
        return nonlinearity_score.unsqueeze(1)

    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)
        if self.training and not self.buffer_filled:
            self.update_training_buffer(x)

        x_centered = x - x.mean(dim=1, keepdim=True)
        x_scaled = x_centered / (torch.norm(x_centered, dim=1, keepdim=True) + self.eps)

        nonlinearity_score = self.compute_nonlinearity_score(x_scaled)
        context_feats = self.context_projector(x_scaled)
        gates = self.gate_generator(context_feats)

        effective_gates = 1.0 - (self.gate_strength * (1.0 - gates) * nonlinearity_score)

        encoded = self.encoder(x_scaled)
        abs_encoded = torch.abs(encoded).float()

        if self.training:
            feature_means = abs_encoded.mean(dim=0)
            threshold = torch.quantile(abs_encoded, 0.8, dim=1, keepdim=True).detach()
            threshold = torch.maximum(threshold, 0.1 * feature_means.view(1, -1))
            soft_temp = torch.exp(-5 * feature_means).view(1, -1) * 0.1
            soft_threshold = torch.sigmoid((abs_encoded - threshold) / soft_temp)

            # Apply gates to modulate features based on context
            activations = encoded * soft_threshold * effective_gates

            noise_mask = activations.abs().mean(dim=1) < self.eps
            if noise_mask.any():
                noise_scale = 0.01 * threshold.mean()
                noise = torch.randn_like(activations[noise_mask]) * noise_scale
                activations[noise_mask] += noise

            self.reinitialize_unstable_features(activations)
        else:
            base_activations = encoded * (abs_encoded > torch.quantile(abs_encoded, 0.8, dim=1, keepdim=True)).float()
            # Apply gates during inference too
            activations = base_activations * effective_gates

        # Decode with gated activations
        decoded = self.decoder(activations)
        output = decoded * torch.norm(x_centered, dim=1, keepdim=True) + x.mean(dim=1, keepdim=True)

        return output, activations, effective_gates


class HierarchicalSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, base_dim: int, comp_dim: int,
                 coherence_penalty: float = 0.1, diversity_weight: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Base level dictionary for individual word semantics
        self.base_encoder = StiefelGrassmannianDictionary(input_dim, base_dim, eps=self.eps)

        # Compositional level dictionary for phrase semantics
        self.comp_encoder = StiefelGrassmannianDictionary(input_dim, comp_dim, eps=self.eps)

        # Combined decoder that reconstructs from both levels
        self.decoder = StiefelGrassmannianDictionary(base_dim + comp_dim, input_dim, eps=self.eps)

        # Router network to determine which level is more appropriate
        self.router = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.activation = nn.ReLU()
        self.coherence_penalty = coherence_penalty
        self.diversity_weight = diversity_weight
        self.base_sparsity_controller = AdaptiveSparsityController(target_sparsity=0.2)
        self.comp_sparsity_controller = AdaptiveSparsityController(
            target_sparsity=0.4)  # Higher sparsity for compositional features

        self.semantic_cache = {}
        self.feature_ema = None
        self.ema_decay = 0.99
        self.buffer_size = 10000
        self.training_buffer = None
        self.buffer_filled = False

    def update_training_buffer(self, x: torch.Tensor):
        if not self.buffer_filled:
            x_float = x.detach().to(torch.float32)
            if self.training_buffer is None:
                self.training_buffer = x_float.clone()
            else:
                self.training_buffer = torch.cat([self.training_buffer, x_float], dim=0)
                if self.training_buffer.size(0) >= self.buffer_size:
                    self.buffer_filled = True
                    self.initialize_pca()

    def initialize_pca(self):
        with torch.no_grad():
            device = self.training_buffer.device
            buffer = self.training_buffer.cpu().float()
            centered = buffer - buffer.mean(dim=0, keepdim=True)
            cov = centered.T @ centered / (centered.size(0) - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvectors = eigenvectors[:, sorted_indices]

            # Initialize base dictionary with principal components
            base_components = eigenvectors[:, :self.base_encoder.dict_size]
            orig_dtype = self.base_encoder.dictionary.weight.dtype
            self.base_encoder.dictionary.weight.data = base_components.T.to(device=device, dtype=orig_dtype)

            # Initialize compositional dictionary with random orthogonal vectors
            self.comp_encoder.initialize_stiefel()

            # Initialize decoder
            comb_components = torch.zeros((self.base_encoder.dict_size + self.comp_encoder.dict_size,
                                           self.decoder.input_dim), device=device, dtype=orig_dtype)
            comb_components[:self.base_encoder.dict_size, :] = base_components.T.to(device=device, dtype=orig_dtype)
            self.decoder.dictionary.weight.data = comb_components

            self.training_buffer = None

    def compute_feature_statistics(self, base_activations: torch.Tensor, comp_activations: torch.Tensor):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with autocast(device_type, enabled=False):
            # Combine activations for statistics
            combined_act = torch.cat([base_activations, comp_activations], dim=1).float()
            batch_size = combined_act.size(0)
            act_norm = combined_act / (torch.norm(combined_act, dim=1, keepdim=True) + self.eps)

            gram = torch.mm(act_norm, act_norm.t())
            mask = torch.triu(torch.ones_like(gram), diagonal=1).bool()
            coherence_mean = torch.abs(gram[mask]).mean()

            cov_matrix = (combined_act.T @ combined_act) / (batch_size - 1)
            eye = torch.eye(cov_matrix.size(0), device=cov_matrix.device)
            eigenvals = torch.linalg.eigvalsh(cov_matrix + self.eps * eye)
            normalized_eigenvals = eigenvals / (torch.sum(eigenvals) + self.eps)
            diversity = 1.0 - torch.max(normalized_eigenvals) / (torch.sum(normalized_eigenvals) + self.eps)

            singular_values = torch.linalg.svdvals(combined_act).float()
            effective_rank = (singular_values / singular_values[0]).sum()

            # Compute compositional utilization - how much the model relies on compositional features
            comp_utilization = torch.norm(comp_activations, dim=1) / (torch.norm(combined_act, dim=1) + self.eps)
            comp_utilization = comp_utilization.mean().item()

            return coherence_mean.item(), diversity.item(), (
                        effective_rank / combined_act.size(1)).item(), comp_utilization

    def track_semantic_consistency(self, combined_activations: torch.Tensor, batch_id: str):
        act_norm = combined_activations / (torch.norm(combined_activations, dim=1, keepdim=True) + self.eps)
        if batch_id in self.semantic_cache:
            prev_norm = self.semantic_cache[batch_id]
            consistency = torch.mean(torch.sum(act_norm * prev_norm, dim=1))
            self.semantic_cache[batch_id] = 0.9 * prev_norm + 0.1 * act_norm.detach()
            return consistency.item()
        self.semantic_cache[batch_id] = act_norm.detach()
        return 1.0

    def compute_complexity_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a score indicating semantic complexity that may require compositional features."""
        # Project to router network
        route_score = self.router(x)
        return route_score

    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)
        if self.training and not self.buffer_filled:
            self.update_training_buffer(x)

        # Center and normalize input
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_scaled = x_centered / (torch.norm(x_centered, dim=1, keepdim=True) + self.eps)

        # Determine routing score
        complexity_score = self.compute_complexity_score(x_scaled)

        # Encode with both dictionaries
        base_encoded = self.base_encoder(x_scaled)
        comp_encoded = self.comp_encoder(x_scaled)

        abs_base = torch.abs(base_encoded).float()
        abs_comp = torch.abs(comp_encoded).float()

        if self.training:
            # Base dictionary activations
            base_means = abs_base.mean(dim=0)
            base_threshold = torch.quantile(abs_base, 0.8, dim=1, keepdim=True).detach()
            base_threshold = torch.maximum(base_threshold, 0.1 * base_means.view(1, -1))
            base_temp = torch.exp(-5 * base_means).view(1, -1) * 0.1
            base_mask = torch.sigmoid((abs_base - base_threshold) / base_temp)
            base_activations = base_encoded * base_mask

            # Compositional dictionary activations - more sparse
            comp_means = abs_comp.mean(dim=0)
            comp_threshold = torch.quantile(abs_comp, 0.9, dim=1, keepdim=True).detach()
            comp_threshold = torch.maximum(comp_threshold, 0.15 * comp_means.view(1, -1))
            comp_temp = torch.exp(-5 * comp_means).view(1, -1) * 0.1
            comp_mask = torch.sigmoid((abs_comp - comp_threshold) / comp_temp)

            # Scale compositional features by complexity score
            comp_activations = comp_encoded * comp_mask * complexity_score

            # Add noise to prevent dead features
            base_noise_mask = base_activations.abs().mean(dim=1) < self.eps
            if base_noise_mask.any():
                noise_scale = 0.01 * base_threshold.mean()
                noise = torch.randn_like(base_activations[base_noise_mask]) * noise_scale
                base_activations[base_noise_mask] += noise

            comp_noise_mask = comp_activations.abs().mean(dim=1) < self.eps
            if comp_noise_mask.any():
                noise_scale = 0.01 * comp_threshold.mean()
                noise = torch.randn_like(comp_activations[comp_noise_mask]) * noise_scale
                comp_activations[comp_noise_mask] += noise
        else:
            # Simpler thresholding for inference
            base_activations = base_encoded * (abs_base > torch.quantile(abs_base, 0.8, dim=1, keepdim=True)).float()
            comp_activations = comp_encoded * (
                        abs_comp > torch.quantile(abs_comp, 0.9, dim=1, keepdim=True)).float() * complexity_score

        # Combine activations from both levels
        combined_activations = torch.cat([base_activations, comp_activations], dim=1)

        # Decode combined activations
        decoded = self.decoder(combined_activations)
        output = decoded * torch.norm(x_centered, dim=1, keepdim=True) + x.mean(dim=1, keepdim=True)

        return output, combined_activations, base_activations, comp_activations, complexity_score

class DynamicLossScaler:
    def __init__(self, initial_scales: Dict[str, float]):
        self.scales = initial_scales
        self.history = {k: [] for k in initial_scales}
        self.ema_beta = 0.98
        self.scale_bounds = {"reconstruction": (0.5, 2.0), "sparsity": (0.05, 0.3), "diversity": (0.02, 0.15), "stability": (0.01, 0.1), "orthogonality": (0.005, 0.05)}

    def update_scales(self, metrics: Dict[str, float]) -> None:
        for key in self.scales:
            if key in metrics:
                self.history[key].append(metrics[key])
                if len(self.history[key]) > 10:
                    trend = sum(x - y for x, y in zip(self.history[key][-5:], self.history[key][-6:-1])) / 5
                    scale_adjustment = 1.0 + 0.1 * np.sign(trend)
                    new_scale = self.scales[key] * scale_adjustment
                    min_bound, max_bound = self.scale_bounds[key]
                    self.scales[key] = max(min_bound, min(max_bound, new_scale))

    def get_scales(self) -> Dict[str, float]:
        return self.scales


class AdaptiveCurriculumTrainer:
    def __init__(self, model_path: str, layer_dims, device: str = "cuda", beta: float = 0.98,
                 model_type: str = "gated"):  # Added model_type parameter
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.hidden_size = self.model.config.hidden_size
        self.model_type = model_type

        if hasattr(self.model, "generation_config"):
            current_temp = getattr(self.model.generation_config, "temperature", 0.6)
            self.model.generation_config.temperature = max(0.5, min(current_temp, 0.7))
        else:
            self.model.generation_config = type("GenerationConfig", (object,), {})()
            self.model.generation_config.temperature = 0.6

        # Initialize appropriate autoencoder model based on model_type
        if model_type == "gated":
            self.saes = nn.ModuleList([
                GatedSparseAutoencoder(self.hidden_size, hidden_dim).to(self.device)
                for _, hidden_dim in layer_dims
            ])
        elif model_type == "hierarchical":
            self.saes = nn.ModuleList([
                HierarchicalSparseAutoencoder(
                    self.hidden_size,
                    base_dim=int(hidden_dim * 0.7),  # 70% for base dictionary
                    comp_dim=int(hidden_dim * 0.3)  # 30% for compositional dictionary
                ).to(self.device)
                for _, hidden_dim in layer_dims
            ])
        else:
            # Original model
            self.saes = nn.ModuleList([
                SparseAutoencoder(self.hidden_size, hidden_dim).to(self.device)
                for _, hidden_dim in layer_dims
            ])

        self.beta = beta
        self.difficulty_history = []
        self.best_metrics = {}
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        wandb.init(project="sdl_steifel", config={
            "layers": layer_dims,
            "model": model_path,
            "beta": beta,
            "model_type": model_type
        })

    def save_checkpoint(self, layer_idx: int, epoch: int, optimizer, scaler) -> None:
        """Save model checkpoint with all necessary state."""
        path = os.path.join(self.checkpoint_dir, f"layer_{layer_idx}_epoch_{epoch}.pt")

        if self.model_type == "gated":
            sae = self.saes[layer_idx]
            checkpoint_data = {
                "model_state": sae.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "metrics": self.best_metrics,
                "feature_ema": sae.feature_ema,
                "semantic_cache": sae.semantic_cache,
                "model_type": self.model_type
            }
            torch.save(checkpoint_data, path)

        elif self.model_type == "hierarchical":
            sae = self.saes[layer_idx]
            checkpoint_data = {
                "model_state": sae.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "metrics": self.best_metrics,
                "feature_ema": sae.feature_ema,
                "semantic_cache": sae.semantic_cache,
                "model_type": self.model_type,
                "compositional_metrics": {"comp_utilization": self.best_metrics.get("comp_utilization", 0.0)}
            }
            torch.save(checkpoint_data, path)

        else:
            # Original model checkpoint saving
            sae = self.saes[layer_idx]
            torch.save({
                "model_state": sae.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "metrics": self.best_metrics,
                "feature_ema": sae.feature_ema,
                "semantic_cache": sae.semantic_cache
            }, path)

    def log_metrics(self, layer_idx: int, epoch: int, train_loss: float, val_metrics: FeatureMetrics,
                    optimizer) -> None:
        """Log training and validation metrics to wandb and console."""
        log_data = {
            f"layer_{layer_idx}/train_loss": train_loss,
            f"layer_{layer_idx}/lr": optimizer.param_groups[0]["lr"]
        }

        # Log validation metrics
        log_data.update({f"layer_{layer_idx}/val_{k}": v for k, v in vars(val_metrics).items()})

        # Log model-specific metrics
        if self.model_type == "gated" and "gate_activity" in self.best_metrics:
            log_data[f"layer_{layer_idx}/gate_activity"] = self.best_metrics["gate_activity"]

        if self.model_type == "hierarchical" and "comp_utilization" in self.best_metrics:
            log_data[f"layer_{layer_idx}/comp_utilization"] = self.best_metrics["comp_utilization"]

        wandb.log(log_data)

        print(f"\nLayer {layer_idx} Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print("Validation Metrics:")
        for key, value in vars(val_metrics).items():
            print(f"{key:>20}: {value:.4f}")

        # Print model-specific metrics
        if self.model_type == "gated" and "gate_activity" in self.best_metrics:
            print(f"{'gate_activity':>20}: {self.best_metrics['gate_activity']:.4f}")

        if self.model_type == "hierarchical" and "comp_utilization" in self.best_metrics:
            print(f"{'comp_utilization':>20}: {self.best_metrics['comp_utilization']:.4f}")

    def train_layer(self, layer_idx: int, train_data, val_data, epochs: int = 50, batch_size: int = 64):
        sae = self.saes[layer_idx]
        optimizer = optim.AdamW(sae.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.999))
        scaler = GradScaler()

        # Configure loss scaler based on model type
        if self.model_type == "gated":
            loss_scaler = DynamicLossScaler({
                "reconstruction": 1.0,
                "sparsity": 0.1,
                "diversity": 0.05,
                "stability": 0.025,
                "orthogonality": 0.01,
                "gating": 0.03  # New loss component for gating mechanism
            })
        elif self.model_type == "hierarchical":
            loss_scaler = DynamicLossScaler({
                "reconstruction": 1.0,
                "sparsity": 0.1,
                "diversity": 0.05,
                "stability": 0.025,
                "orthogonality": 0.01,
                "composition": 0.05  # New loss component for compositional features
            })
        else:
            loss_scaler = DynamicLossScaler({
                "reconstruction": 1.0,
                "sparsity": 0.1,
                "diversity": 0.05,
                "stability": 0.025,
                "orthogonality": 0.01
            })

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=epochs,
            steps_per_epoch=len(train_data), pct_start=0.3,
            anneal_strategy='cos'
        )

        grad_accum_steps = max(1, 256 // batch_size)
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16

        for epoch in range(epochs):
            sae.train()
            epoch_metrics = defaultdict(float)
            optimizer.zero_grad(set_to_none=True)
            progress_bar = tqdm(train_data, desc=f"Layer {layer_idx} Epoch {epoch}")

            # Update sparsity target
            if self.model_type == "hierarchical":
                sae.base_sparsity_controller.update_target(epoch, epochs)
                sae.comp_sparsity_controller.update_target(epoch, epochs)
            else:
                sae.sparsity_controller.update_target(epoch, epochs)

            for batch_idx, batch in enumerate(progress_bar):
                x = batch.to(self.device, non_blocking=True)
                with torch.no_grad(), autocast(self.device, dtype=dtype):
                    hidden = self.model(x, output_hidden_states=True).hidden_states[layer_idx]
                    hidden = hidden.view(-1, self.hidden_size)

                with autocast(self.device, dtype=dtype):
                    # Forward pass depends on model type
                    if self.model_type == "gated":
                        recon, activations, gates = sae(hidden)
                        recon_loss = F.mse_loss(recon, hidden)
                        sparsity = (activations.abs() < 0.01).float().mean()
                        sparsity_penalty = sae.sparsity_controller.compute_penalty(sparsity.item())
                        frame_potential = sae.encoder.compute_frame_potential()
                        coherence, diversity, stability = sae.compute_feature_statistics(activations)
                        ortho_loss = sae.decoder.compute_frame_potential()

                        # Add gate regularization to prevent collapse
                        gate_diversity_loss = -torch.var(gates)

                        scales = loss_scaler.get_scales()
                        loss = (
                                       scales["reconstruction"] * recon_loss +
                                       scales["sparsity"] * sparsity_penalty * frame_potential +
                                       scales["diversity"] * (1 - diversity) +
                                       scales["stability"] * (1 - stability) +
                                       scales["orthogonality"] * ortho_loss +
                                       scales["gating"] * gate_diversity_loss
                               ) / grad_accum_steps

                        metrics = {
                            "loss": loss.item() * grad_accum_steps,
                            "sparsity": sparsity.item(),
                            "diversity": diversity,
                            "stability": stability,
                            "recon": recon_loss.item(),
                            "ortho": ortho_loss.item(),
                            "gate_div": gate_diversity_loss.item()
                        }

                    elif self.model_type == "hierarchical":
                        recon, combined_activations, base_activations, comp_activations, complexity_score = sae(hidden)
                        recon_loss = F.mse_loss(recon, hidden)

                        # Compute statistics for both dictionaries
                        base_sparsity = (base_activations.abs() < 0.01).float().mean()
                        comp_sparsity = (comp_activations.abs() < 0.01).float().mean()

                        base_penalty = sae.base_sparsity_controller.compute_penalty(base_sparsity.item())
                        comp_penalty = sae.comp_sparsity_controller.compute_penalty(comp_sparsity.item())

                        base_frame_potential = sae.base_encoder.compute_frame_potential()
                        comp_frame_potential = sae.comp_encoder.compute_frame_potential()

                        coherence, diversity, stability, comp_util = sae.compute_feature_statistics(
                            base_activations, comp_activations
                        )

                        ortho_loss = sae.decoder.compute_frame_potential()

                        # Compositional regularization to encourage proper usage
                        comp_reg_loss = F.mse_loss(
                            complexity_score,
                            torch.sigmoid(torch.sum(comp_activations.abs(), dim=1, keepdim=True))
                        )

                        scales = loss_scaler.get_scales()
                        loss = (
                                       scales["reconstruction"] * recon_loss +
                                       scales["sparsity"] * (base_penalty * base_frame_potential +
                                                             comp_penalty * comp_frame_potential) +
                                       scales["diversity"] * (1 - diversity) +
                                       scales["stability"] * (1 - stability) +
                                       scales["orthogonality"] * ortho_loss +
                                       scales["composition"] * comp_reg_loss
                               ) / grad_accum_steps

                        metrics = {
                            "loss": loss.item() * grad_accum_steps,
                            "sparsity": (base_sparsity.item() + comp_sparsity.item()) / 2,
                            "diversity": diversity,
                            "stability": stability,
                            "recon": recon_loss.item(),
                            "ortho": ortho_loss.item(),
                            "comp_util": comp_util
                        }

                    else:
                        # Original model forward pass
                        recon, activations = sae(hidden)
                        recon_loss = F.mse_loss(recon, hidden)
                        sparsity = (activations.abs() < 0.01).float().mean()
                        sparsity_penalty = sae.sparsity_controller.compute_penalty(sparsity.item())
                        frame_potential = sae.encoder.compute_frame_potential()
                        coherence, diversity, stability = sae.compute_feature_statistics(activations)
                        ortho_loss = sae.decoder.compute_frame_potential()

                        scales = loss_scaler.get_scales()
                        loss = (
                                       scales["reconstruction"] * recon_loss +
                                       scales["sparsity"] * sparsity_penalty * frame_potential +
                                       scales["diversity"] * (1 - diversity) +
                                       scales["stability"] * (1 - stability) +
                                       scales["orthogonality"] * ortho_loss
                               ) / grad_accum_steps

                        metrics = {
                            "loss": loss.item() * grad_accum_steps,
                            "sparsity": sparsity.item(),
                            "diversity": diversity,
                            "stability": stability,
                            "recon": recon_loss.item(),
                            "ortho": ortho_loss.item()
                        }

                scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    # Maintain orthogonality constraints
                    if self.model_type == "hierarchical":
                        sae.base_encoder.project_to_stiefel()
                        sae.comp_encoder.project_to_stiefel()
                        sae.decoder.project_to_stiefel()
                    else:
                        sae.encoder.project_to_stiefel()
                        sae.decoder.project_to_stiefel()

                for k, v in metrics.items():
                    epoch_metrics[k] += v
                progress_bar.set_postfix(epoch_metrics)

            avg_metrics = {k: v / len(train_data) for k, v in epoch_metrics.items()}
            loss_scaler.update_scales(avg_metrics)
            val_metrics = self.validate(layer_idx, val_data)
            self.save_checkpoint(layer_idx, epoch, optimizer, scaler)
            self.log_metrics(layer_idx, epoch, avg_metrics["loss"], val_metrics, optimizer)

    def validate(self, layer_idx: int, val_data) -> FeatureMetrics:
        sae = self.saes[layer_idx]
        sae.eval()

        if self.model_type == "hierarchical":
            total = FeatureMetrics(0, 0, 0, 0, 0, 0, 0)
            comp_utilization = 0.0

            with torch.no_grad():
                for batch in val_data:
                    x = batch.to(self.device)
                    hidden = self.model(x, output_hidden_states=True).hidden_states[layer_idx]
                    hidden = hidden.view(-1, self.hidden_size).float()

                    recon, combined_activations, base_activations, comp_activations, complexity_score = sae(hidden)
                    recon_loss = nn.MSELoss()(recon, hidden)

                    # Combined sparsity
                    sparsity = ((combined_activations.abs() < 0.01).float().mean() +
                                (base_activations.abs() < 0.01).float().mean() +
                                (comp_activations.abs() < 0.01).float().mean()) / 3.0

                    base_frame_potential = sae.base_encoder.compute_frame_potential()
                    comp_frame_potential = sae.comp_encoder.compute_frame_potential()
                    frame_potential = (base_frame_potential + comp_frame_potential) / 2.0

                    coherence, diversity, stability, comp_util = sae.compute_feature_statistics(
                        base_activations, comp_activations
                    )

                    semantic = sae.track_semantic_consistency(combined_activations, "val")
                    ortho_loss = sae.decoder.compute_frame_potential()

                    batch_metrics = FeatureMetrics(
                        float(sparsity.item()),
                        float(coherence),
                        float(stability),
                        float(diversity),
                        float(semantic),
                        float(recon_loss.item()),
                        float(ortho_loss.item())
                    )

                    comp_utilization += comp_util

                    for field in FeatureMetrics.__dataclass_fields__:
                        setattr(total, field, getattr(total, field) + getattr(batch_metrics, field))

            num_batches = len(val_data)
            for field in FeatureMetrics.__dataclass_fields__:
                setattr(total, field, getattr(total, field) / num_batches)

            # Add compositional utilization to logging
            self.best_metrics["comp_utilization"] = comp_utilization / num_batches

            return total

        elif self.model_type == "gated":
            total = FeatureMetrics(0, 0, 0, 0, 0, 0, 0)
            gate_activity = 0.0

            with torch.no_grad():
                for batch in val_data:
                    x = batch.to(self.device)
                    hidden = self.model(x, output_hidden_states=True).hidden_states[layer_idx]
                    hidden = hidden.view(-1, self.hidden_size).float()

                    recon, activations, gates = sae(hidden)
                    recon_loss = nn.MSELoss()(recon, hidden)

                    sparsity = (activations.abs() < 0.01).float().mean()
                    frame_potential = sae.encoder.compute_frame_potential()
                    coherence, diversity, stability = sae.compute_feature_statistics(activations)
                    semantic = sae.track_semantic_consistency(activations, "val")
                    ortho_loss = sae.decoder.compute_frame_potential()

                    # Track gate activity to measure nonlinearity detection
                    gate_activity += (1.0 - gates.mean()).item()

                    batch_metrics = FeatureMetrics(
                        float(sparsity.item()),
                        float(coherence),
                        float(stability),
                        float(diversity),
                        float(semantic),
                        float(recon_loss.item()),
                        float(ortho_loss.item())
                    )

                    for field in FeatureMetrics.__dataclass_fields__:
                        setattr(total, field, getattr(total, field) + getattr(batch_metrics, field))

            num_batches = len(val_data)
            for field in FeatureMetrics.__dataclass_fields__:
                setattr(total, field, getattr(total, field) / num_batches)

            # Add gate activity to logging
            self.best_metrics["gate_activity"] = gate_activity / num_batches

            return total

        else:
            # Original implementation
            total = FeatureMetrics(0, 0, 0, 0, 0, 0, 0)
            with torch.no_grad():
                for batch in val_data:
                    x = batch.to(self.device)
                    hidden = self.model(x, output_hidden_states=True).hidden_states[layer_idx]
                    hidden = hidden.view(-1, self.hidden_size).float()
                    recon, activations = sae(hidden)
                    recon_loss = nn.MSELoss()(recon, hidden)
                    sparsity = (activations.abs() < 0.01).float().mean()
                    frame_potential = sae.encoder.compute_frame_potential()
                    coherence, diversity, stability = sae.compute_feature_statistics(activations)
                    semantic = sae.track_semantic_consistency(activations, "val")
                    ortho_loss = sae.decoder.compute_frame_potential()
                    batch_metrics = FeatureMetrics(float(sparsity.item()), float(coherence), float(stability),
                                                   float(diversity), float(semantic), float(recon_loss.item()),
                                                   float(ortho_loss.item()))
                    for field in FeatureMetrics.__dataclass_fields__:
                        setattr(total, field, getattr(total, field) + getattr(batch_metrics, field))
            num_batches = len(val_data)
            for field in FeatureMetrics.__dataclass_fields__:
                setattr(total, field, getattr(total, field) / num_batches)
            return total

    def train_all(self, train_data, val_data, epochs: int = 50, batch_size: int = 64):
        """Train all layers sequentially."""
        for layer_idx in range(len(self.saes)):
            print(f"\n=== Training Layer {layer_idx} ===\n")
            self.train_layer(layer_idx, train_data, val_data, epochs, batch_size)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Choose the model type: "original", "gated", or "hierarchical"
    model_type = "gated"  # Change this to select model type

    print(f"Training using {model_type} sparse autoencoder model")

    trainer = AdaptiveCurriculumTrainer(
        model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        layer_dims=[(None, 2048), (None, 4096)],
        device=device,
        model_type=model_type
    )

    print("Model config:", trainer.model.config)
    print("Hidden size:", trainer.model.config.hidden_size)
    train_data = [torch.randint(0, 10000, (32, 512)).to(device) for _ in range(100)]
    val_data = [torch.randint(0, 10000, (16, 512)).to(device) for _ in range(20)]
    trainer.train_all(train_data, val_data, epochs=50)