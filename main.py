import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.nn.utils.parametrizations import orthogonal
from torch.linalg import qr
from tqdm import tqdm
import numpy as np
import wandb
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split, Subset

# Enable TF32 for faster computations on Ampere GPUs
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
    nonlinearity_score: float = 0.0


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

        # Smoother penalty calculation with clamping for stability
        penalty = torch.tanh(torch.tensor(error * 5))
        return torch.clamp(
            self.min_penalty + (self.max_penalty - self.min_penalty) * (penalty + 1) / 2,
            self.min_penalty,
            self.max_penalty
        )

    def update_target(self, epoch: int, total_epochs: int) -> None:
        # Cosine schedule for target sparsity
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
            Q, _ = torch.linalg.qr(self.dictionary.weight.t(), mode='reduced')
            self.dictionary.weight.data = Q.t()

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


class ManifoldAwareAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, coherence_penalty: float = 0.1, diversity_weight: float = 0.1,
                 eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = StiefelGrassmannianDictionary(input_dim, hidden_dim, eps=self.eps)
        self.decoder = StiefelGrassmannianDictionary(hidden_dim, input_dim, eps=self.eps)
        self.context_projector = nn.Linear(input_dim, hidden_dim // 2)
        self.gate_generator = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim), nn.Sigmoid())

        # Nonlinearity detection components
        self.nonlin_detector_heads = 3
        self.signal_weights = nn.Parameter(torch.ones(self.nonlin_detector_heads) / self.nonlin_detector_heads)
        self.negation_detector = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.manifold_pattern_detector = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.negation_patterns = nn.Parameter(torch.randn(5, input_dim) / (input_dim ** 0.5), requires_grad=True)
        self.gate_strength = nn.Parameter(torch.ones(1) * 0.5)

        # Loss weights
        self.gate_diversity_weight = 0.05
        self.coherence_penalty = coherence_penalty
        self.diversity_weight = diversity_weight

        # Control components
        self.sparsity_controller = AdaptiveSparsityController()

        # Caching and state tracking
        self.buffer_size = 2048
        self.training_buffer = None
        self.buffer_filled = False
        self.semantic_cache = {}
        self.feature_ema = None
        self.feature_ema_var = None
        self.ema_decay = 0.99
        self.last_gates = None
        self.last_nonlinearity_score = None
        self._cached_stats_count = 0

    def update_training_buffer(self, x: torch.Tensor):
        if not self.buffer_filled:
            reduced_buffer_size = min(2048, self.buffer_size)

            if x.dtype not in (torch.float16, torch.bfloat16):
                x_float = x.detach().to(torch.float16 if x.device.type == 'cuda' else torch.bfloat16)
            else:
                x_float = x.detach()

            if self.training_buffer is None:
                self.training_buffer = x_float.clone()
            else:
                if self.training_buffer.size(0) < reduced_buffer_size:
                    if x_float.size(0) > 64:
                        indices = torch.randperm(x_float.size(0), device=x_float.device)[:64]
                        x_sample = x_float[indices]
                    else:
                        x_sample = x_float

                    self.training_buffer = torch.cat([self.training_buffer, x_sample], dim=0)

                    if self.training_buffer.size(0) >= reduced_buffer_size:
                        self.buffer_filled = True
                        self.initialize_pca()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    def initialize_pca(self):
        with torch.no_grad():
            device = self.training_buffer.device
            buffer_size = self.training_buffer.size(0)

            chunk_size = min(1024, buffer_size)
            dtype = torch.float32

            sum_vector = torch.zeros(self.input_dim, dtype=dtype, device='cpu')
            for i in range(0, buffer_size, chunk_size):
                end_idx = min(i + chunk_size, buffer_size)
                chunk = self.training_buffer[i:end_idx].cpu().to(dtype)
                sum_vector += chunk.sum(dim=0)
            mean_vector = sum_vector / buffer_size

            cov = torch.zeros((self.input_dim, self.input_dim), dtype=dtype, device='cpu')
            for i in range(0, buffer_size, chunk_size):
                end_idx = min(i + chunk_size, buffer_size)
                chunk = self.training_buffer[i:end_idx].cpu().to(dtype)
                centered_chunk = chunk - mean_vector.unsqueeze(0)
                cov += centered_chunk.T @ centered_chunk

            cov /= (buffer_size - 1)

            k = min(self.encoder.dict_size + 10, self.input_dim)
            if hasattr(torch.linalg, 'eigvalsh_eigvecs'):
                eigenvalues, eigenvectors = torch.linalg.eigvalsh_eigvecs(cov, UPLO='U',
                                                                          eigvals=(
                                                                          self.input_dim - k, self.input_dim - 1))
            else:
                eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO='U')
                eigenvalues = eigenvalues[-k:]
                eigenvectors = eigenvectors[:, -k:]

            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvectors = eigenvectors[:, sorted_indices]

            components = eigenvectors[:, :self.encoder.dict_size]

            orig_dtype = self.encoder.dictionary.weight.dtype
            self.encoder.dictionary.weight.data = components.T.to(device=device, dtype=orig_dtype)
            self.decoder.dictionary.weight.data = components.to(device=device, dtype=orig_dtype)

            del self.training_buffer
            self.training_buffer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def compute_nonlinearity_score(self, x: torch.Tensor) -> torch.Tensor:
        with autocast(device_type="cuda" if x.is_cuda else "cpu", enabled=True):
            context_feats = self.context_projector(x)

            context_feats_norm = torch.norm(context_feats, dim=1, keepdim=True) + self.eps
            context_norm = context_feats.div(context_feats_norm)

            batch_size = context_norm.size(0)
            if batch_size <= 256:
                sim_matrix = torch.mm(context_norm, context_norm.t())
            else:
                sim_matrix = torch.zeros(batch_size, batch_size, dtype=context_norm.dtype, device=context_norm.device)
                chunk_size = 128
                for i in range(0, batch_size, chunk_size):
                    end_i = min(i + chunk_size, batch_size)
                    for j in range(0, batch_size, chunk_size):
                        end_j = min(j + chunk_size, batch_size)
                        sim_matrix[i:end_i, j:end_j] = torch.mm(context_norm[i:end_i], context_norm[j:end_j].t())

            row_mean = torch.mean(sim_matrix, dim=1, keepdim=True)
            centered = sim_matrix - row_mean
            row_variance = torch.mean(centered ** 2, dim=1) + 1e-8

            sim_scaled = (sim_matrix + 1) / 2
            sim_scaled = torch.clamp(sim_scaled, 1e-6, 1 - 1e-6)
            entropy = -torch.sum(sim_scaled * torch.log(sim_scaled), dim=1)

            max_entropy = torch.log(torch.tensor(batch_size, dtype=torch.float, device=entropy.device))
            norm_entropy = entropy / (max_entropy + self.eps)

            # Compute skewness safely without quantile
            centered_cubed = centered ** 3

            # Use mean absolute deviation instead of quantile for outlier handling
            mean_abs_dev = torch.mean(torch.abs(centered_cubed - centered_cubed.mean()), dim=1)

            # Winsorize using MAD for robustness (safer than quantile for large tensors)
            threshold = 3.0 * mean_abs_dev.unsqueeze(1)  # ~3 MAD threshold
            centered_cubed_clipped = torch.clamp(
                centered_cubed,
                -threshold.expand_as(centered_cubed),
                threshold.expand_as(centered_cubed)
            )

            skewness = torch.mean(centered_cubed_clipped, dim=1) / (row_variance ** 1.5 + 1e-8)

            manifold_score = self.manifold_pattern_detector(x).squeeze(-1)

            signal_weights_norm = F.softmax(self.signal_weights, dim=0)

            # Z-score normalize components for consistent scaling and avoid division by zero
            rv_mean, rv_std = row_variance.mean(), row_variance.std() + 1e-8
            ne_mean, ne_std = norm_entropy.mean(), norm_entropy.std() + 1e-8
            sk_mean, sk_std = torch.abs(skewness).mean(), torch.abs(skewness).std() + 1e-8

            rv_z = (row_variance - rv_mean) / rv_std
            ne_z = (norm_entropy - ne_mean) / ne_std
            sk_z = (torch.abs(skewness) - sk_mean) / sk_std

            # Apply tanh to bound the signal components
            combined_signal = (
                    signal_weights_norm[0] * torch.tanh(rv_z) +
                    signal_weights_norm[1] * torch.tanh(ne_z) +
                    signal_weights_norm[2] * torch.tanh(sk_z)
            )

            nonlinearity_score = torch.sigmoid(combined_signal)

            # Combine with manifold score using LogSumExp (stable maximum)
            temp = 2.0
            stacked = torch.stack([nonlinearity_score * temp, manifold_score * temp], dim=1)
            max_vals, _ = torch.max(stacked, dim=1)
            exp_diff = torch.exp(stacked - max_vals.unsqueeze(1))
            boosted_score = max_vals + torch.log(torch.sum(exp_diff, dim=1)) / temp

            # Ensure values are properly bounded
            return torch.clamp(boosted_score, 0.1, 0.9).unsqueeze(1)

    def detect_negation(self, x: torch.Tensor) -> torch.Tensor:
        direct_score = self.negation_detector(x)

        pattern_scores = []
        for pattern in self.negation_patterns:
            pattern_norm = pattern / (torch.norm(pattern) + self.eps)
            x_norm = x / (torch.norm(x, dim=1, keepdim=True) + self.eps)
            similarity = torch.abs(torch.matmul(x_norm, pattern_norm))
            pattern_scores.append(similarity.unsqueeze(1))

        pattern_score = torch.max(torch.cat(pattern_scores, dim=1), dim=1)[0].unsqueeze(1)
        return torch.max(direct_score, pattern_score)

    def reinitialize_unstable_features(self, activations: torch.Tensor, stability_threshold: float = 0.2):
        with torch.no_grad():
            if self.feature_ema is None:
                self.feature_ema = activations.mean(dim=0).detach().clone()
                self.feature_ema_var = torch.var(activations, dim=0).detach().clamp(min=1e-8)
                return

            # Update EMA with more stable computation
            current_means = activations.mean(dim=0).detach()
            current_var = torch.var(activations, dim=0).detach().clamp(min=1e-8)

            # Update EMAs with exponential moving average
            self.feature_ema = self.ema_decay * self.feature_ema + (1 - self.ema_decay) * current_means
            self.feature_ema_var = self.ema_decay * self.feature_ema_var + (1 - self.ema_decay) * current_var

            # Calculate correlation matrix to detect redundant features
            act_norm = activations / (torch.norm(activations, dim=1, keepdim=True) + self.eps)
            feature_corr = torch.mm(act_norm.T, act_norm)
            diag_mask = ~torch.eye(feature_corr.size(0), dtype=torch.bool, device=feature_corr.device)
            corr_mask = (feature_corr > 0.9) & diag_mask

            # Check stability of each feature
            for i in range(activations.size(1)):
                # Calculate coefficient of variation as stability metric
                feat_mean = self.feature_ema[i]
                feat_std = torch.sqrt(self.feature_ema_var[i])
                cv = feat_std / (torch.abs(feat_mean) + self.eps)
                stability = 1.0 / (1.0 + cv)

                has_correlations = corr_mask[i].any()

                if stability < stability_threshold or has_correlations:
                    # Generate orthogonal vector using modified Gram-Schmidt
                    rand_vec = torch.randn(self.encoder.input_dim, device=activations.device)

                    # Orthogonalize against existing features
                    for j in range(i):
                        basis_vec = self.encoder.dictionary.weight.data[j]
                        basis_norm = torch.norm(basis_vec) + self.eps
                        basis_vec_norm = basis_vec / basis_norm
                        proj = torch.dot(rand_vec, basis_vec_norm)
                        rand_vec = rand_vec - proj * basis_vec_norm

                        # Renormalize for stability after each projection
                        rand_norm = torch.norm(rand_vec)
                        if rand_norm > 1e-6:
                            rand_vec = rand_vec / rand_norm
                        else:
                            # Restart if we get numerical collapse
                            rand_vec = torch.randn(self.encoder.input_dim, device=activations.device)
                            break

                    # Final normalization
                    rand_vec = rand_vec / (torch.norm(rand_vec) + self.eps)

                    # Update dictionary elements
                    self.encoder.dictionary.weight.data[i] = rand_vec
                    if i < self.decoder.input_dim:
                        self.decoder.dictionary.weight.data[:, i] = rand_vec

    def compute_feature_statistics(self, activations: torch.Tensor):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        # Fast path for efficiency - return cached values when possible
        if hasattr(self, '_cached_stats_count') and self._cached_stats_count > 0:
            self._cached_stats_count -= 1
            return (
                getattr(self, '_cached_coherence', 0.2),
                getattr(self, '_cached_diversity', 0.7),
                getattr(self, '_cached_stability', 0.8),
                self.last_nonlinearity_score.mean().item() if self.last_nonlinearity_score is not None else 0.5,
                (1.0 - self.last_gates).mean().item() if hasattr(self, 'last_gates') else 0.5
            )

        # Compute actual statistics with numerical stability enhancements
        with torch.no_grad(), autocast(device_type, enabled=False):
            # Use precision appropriate for statistics
            compute_dtype = torch.float32

            # Sample for efficiency (at most 128 vectors)
            n_samples = min(128, activations.size(0))
            if activations.size(0) > n_samples:
                indices = torch.randperm(activations.size(0), device=activations.device)[:n_samples]
                act = activations[indices].to(compute_dtype)
            else:
                act = activations.to(compute_dtype)

            # Coherence calculation using mutual coherence from compressed sensing
            if act.size(0) >= 4:
                act_norm = F.normalize(act, p=2, dim=1)
                gram = torch.mm(act_norm, act_norm.t())

                # Extract upper triangle excluding diagonal
                mask = torch.triu(torch.ones_like(gram), diagonal=1).bool()
                if mask.any():
                    # Use median rather than mean for robustness
                    coherence_vals = torch.abs(gram[mask])
                    coherence_mean = torch.median(coherence_vals).item()
                else:
                    coherence_mean = 0.5
            else:
                coherence_mean = 0.5

            # Diversity calculation using eigenvalue distribution
            if act.size(0) >= 4:
                # Center data for proper analysis
                centered = act - act.mean(dim=0, keepdim=True)

                # Use SVD for numerical stability
                try:
                    _, S, _ = torch.svd(centered, compute_uv=True, some=True)
                    var_explained = (S ** 2) / (act.size(0) - 1 + self.eps)

                    # Calculate normalized variance explained
                    total_var = var_explained.sum() + self.eps
                    prop_var = var_explained / total_var

                    # Use Gini coefficient for diversity - more stable than entropy
                    sorted_var, _ = torch.sort(prop_var)
                    n = prop_var.size(0)
                    idx = torch.arange(1, n + 1, device=prop_var.device)
                    cumsum = torch.cumsum(sorted_var, dim=0)
                    gini = (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
                    diversity = gini.item()
                except:
                    # Fallback to simpler computation if SVD fails
                    feat_var = torch.var(act, dim=0) + self.eps
                    sorted_var, _ = torch.sort(feat_var, descending=True)
                    top_k = min(32, len(sorted_var))
                    top_sum = sorted_var[:top_k].sum()
                    diversity = (1.0 - (sorted_var[0] / top_sum)).item() if top_sum > 0 else 0.5
            else:
                diversity = 0.7

            # Stability calculation
            if self.feature_ema is not None:
                current_means = act.mean(dim=0)
                # Calculate relative change normalized by standard deviation
                sd = torch.sqrt(self.feature_ema_var + self.eps)
                rel_change = torch.abs(current_means - self.feature_ema) / (sd + self.eps)
                # Convert to stability score via sigmoid
                stability = torch.sigmoid(-rel_change.mean() * 5 + 2).item()
            else:
                stability = 0.8

            # Nonlinearity score - use cached value with small random variation
            nonlinearity = self.last_nonlinearity_score.mean().item() if self.last_nonlinearity_score is not None else 0.5

            # Gate utilization
            gate_util = (1.0 - self.last_gates).mean().item() if hasattr(self, 'last_gates') else 0.5

            # Add small random variation to prevent numerical stagnation
            if self.training:
                jitter = 0.02
                coherence_mean *= (1.0 - jitter + 2 * jitter * torch.rand(1).item())
                diversity *= (1.0 - jitter + 2 * jitter * torch.rand(1).item())
                stability *= (1.0 - jitter + 2 * jitter * torch.rand(1).item())
                nonlinearity *= (1.0 - jitter + 2 * jitter * torch.rand(1).item())

            # Clamp to reasonable ranges
            coherence_mean = max(0.1, min(0.95, coherence_mean))
            diversity = max(0.1, min(0.95, diversity))
            stability = max(0.1, min(0.95, stability))
            nonlinearity = max(0.1, min(0.95, nonlinearity))

        # Cache computed values
        self._cached_coherence = coherence_mean
        self._cached_diversity = diversity
        self._cached_stability = stability
        self._cached_stats_count = 5

        return coherence_mean, diversity, stability, nonlinearity, gate_util

    def track_semantic_consistency(self, activations: torch.Tensor, batch_id: str):
        act_norm = activations / (torch.norm(activations, dim=1, keepdim=True) + self.eps)
        if batch_id in self.semantic_cache:
            prev_norm = self.semantic_cache[batch_id]
            consistency = torch.mean(torch.sum(act_norm * prev_norm, dim=1))
            self.semantic_cache[batch_id] = 0.9 * prev_norm + 0.1 * act_norm.detach()
            return consistency.item()

        self.semantic_cache[batch_id] = act_norm.detach()
        return 1.0

    def compute_gate_diversity_loss(self, gates: torch.Tensor) -> torch.Tensor:
        mean_activations = gates.mean(dim=0)
        target_activation = 0.5
        centrality_loss = ((mean_activations - target_activation) ** 2).mean()
        variance_loss = -torch.var(gates, dim=0).mean()

        gates_centered = gates - gates.mean(dim=0, keepdim=True)
        covariance = (gates_centered.t() @ gates_centered) / (gates.size(0) - 1 + self.eps)

        off_diag_cov = covariance.flatten()[:-1].view(gates.size(1) - 1, gates.size(1) + 1)[:, 1:].flatten()
        correlation_loss = torch.mean(torch.abs(off_diag_cov))

        return centrality_loss + variance_loss + correlation_loss

    def forward(self, x: torch.Tensor, batch_idx=None):
        x = x.to(torch.float32)
        if self.training and not self.buffer_filled:
            self.update_training_buffer(x)

        # Normalize with epsilon terms for stability
        x_norm = torch.norm(x, dim=1, keepdim=True) + self.eps
        x_mean = x.mean(dim=1, keepdim=True)
        x_centered = x - x_mean
        x_scaled = x_centered / x_norm

        # Context features
        context_feats = self.context_projector(x_scaled)

        # Compute nonlinearity less often for speed
        should_compute_nonlin = batch_idx is None or batch_idx % 10 == 0
        if should_compute_nonlin:
            nonlinearity_score = self.compute_nonlinearity_score(x_scaled)
            self._cached_nonlinearity = nonlinearity_score.detach()
        else:
            # Reuse cached value with small jitter for stability
            nonlinearity_score = getattr(self, '_cached_nonlinearity',
                                         torch.ones((x.size(0), 1), device=x.device) * 0.5)
            if self.training:
                nonlinearity_score = nonlinearity_score * (0.98 + 0.04 * torch.rand(1, device=x.device).item())

        # Compute negation rarely for efficiency
        if should_compute_nonlin and (batch_idx is None or batch_idx % 20 == 0):
            negation_score = self.detect_negation(x_scaled)
            combined_score = torch.max(nonlinearity_score, negation_score)
        else:
            combined_score = nonlinearity_score

        # Store for metrics
        self.last_nonlinearity_score = combined_score.detach()

        # Generate gates
        gates = self.gate_generator(context_feats)

        # Apply noise during training for regularization
        if self.training and (batch_idx is None or batch_idx % 5 == 0):
            noise_scale = 0.01 * (1.0 - 0.9 * min(1.0, batch_idx / 1000 if batch_idx is not None else 0))
            gate_noise = torch.randn_like(gates) * noise_scale
            gates = torch.clamp(gates + gate_noise, 0.01, 0.99)

        # Calculate effective gates with bounded strength
        gate_strength = torch.sigmoid(self.gate_strength)
        effective_gates = 1.0 - (gate_strength * (1.0 - gates) * combined_score)
        self.last_gates = effective_gates.detach().mean(dim=0)

        # Forward through encoder
        encoded = self.encoder(x_scaled)

        # Fast-path for early training
        if self.training and batch_idx is not None and batch_idx < 25:
            abs_encoded = torch.abs(encoded).float()

            # Robust threshold based on mean + std
            thresh_mean = abs_encoded.mean(dim=0, keepdim=True)
            thresh_std = abs_encoded.std(dim=0, keepdim=True) + 1e-8
            threshold = thresh_mean + 0.5 * thresh_std

            # Apply threshold and gates
            activations = encoded * (abs_encoded > threshold).to(encoded.dtype) * effective_gates
            decoded = self.decoder(activations)
            output = decoded * x_norm + x_mean
            return output, activations, effective_gates

        # Full processing path
        abs_encoded = torch.abs(encoded).float()

        if self.training:
            # Calculate feature statistics
            feature_means = abs_encoded.mean(dim=0) + 1e-10
            feature_stds = abs_encoded.std(dim=0) + 1e-10

            # Alternative to quantile for threshold calculation
            if abs_encoded.size(0) > 100:
                # For large batches, use mean + k*std as threshold (faster than quantile)
                k = 1.0  # ~84th percentile assuming normal distribution
                threshold = abs_encoded.mean(dim=1, keepdim=True) + k * abs_encoded.std(dim=1, keepdim=True)
            else:
                # For small batches, sort and pick kth largest value
                k = max(1, int(abs_encoded.size(1) * 0.2))  # 80th percentile
                sorted_vals, _ = torch.sort(abs_encoded, dim=1, descending=True)
                threshold = sorted_vals[:, k:k + 1]

            threshold = threshold.detach()

            # Ensure reasonable minimum threshold based on feature means
            min_threshold = 0.1 * feature_means.view(1, -1)
            threshold = torch.maximum(threshold, min_threshold)

            # Smooth thresholding with sigmoid for numerical stability
            input_scaled = (abs_encoded - threshold) * 10
            soft_threshold = 1.0 / (1.0 + torch.exp(-input_scaled))

            # Apply soft threshold and gates
            activations = encoded * soft_threshold * effective_gates

            # Prevent dead features
            noise_mask = (activations.abs().mean(dim=1) < self.eps)
            if noise_mask.any():
                noise_scale = 0.01 * threshold.mean()
                noise = torch.randn_like(activations[noise_mask]) * noise_scale
                activations[noise_mask] += noise

            # Periodically reinitialize unstable features
            if batch_idx is not None and batch_idx % 100 == 0:
                self.reinitialize_unstable_features(activations)
        else:
            # Evaluation mode - use sorted values as threshold
            if abs_encoded.size(0) > 100:
                threshold = abs_encoded.mean(dim=1, keepdim=True) + abs_encoded.std(dim=1, keepdim=True)
            else:
                k = max(1, int(abs_encoded.size(1) * 0.2))
                sorted_vals, _ = torch.sort(abs_encoded, dim=1, descending=True)
                threshold = sorted_vals[:, k:k + 1]

            # Ensure minimum threshold
            mean_thresh = torch.mean(abs_encoded, dim=1, keepdim=True) * 0.5
            threshold = torch.maximum(threshold, mean_thresh)

            # Apply threshold and gates
            base_activations = encoded * (abs_encoded > threshold).to(encoded.dtype)
            activations = base_activations * effective_gates

        # Decode with original scaling
        decoded = self.decoder(activations)
        output = decoded * x_norm + x_mean

        return output, activations, effective_gates


class DynamicLossScaler:
    def __init__(self, initial_scales: Dict[str, float]):
        self.scales = initial_scales
        self.history = {k: [] for k in initial_scales}
        self.ema_beta = 0.98
        self.scale_bounds = {
            "reconstruction": (0.5, 2.0),
            "sparsity": (0.05, 0.3),
            "diversity": (0.02, 0.15),
            "stability": (0.01, 0.1),
            "orthogonality": (0.005, 0.05),
            "nonlinearity": (0.01, 0.2),
            "gate_diversity": (0.01, 0.15)
        }

    def update_scales(self, metrics: Dict[str, float]) -> None:
        for key in self.scales:
            if key in metrics:
                self.history[key].append(metrics[key])
                if len(self.history[key]) > 10:
                    # Compute trend over last 5 iterations
                    trend = sum(x - y for x, y in zip(self.history[key][-5:], self.history[key][-6:-1])) / 5
                    scale_adjustment = 1.0 + 0.1 * np.sign(trend)
                    new_scale = self.scales[key] * scale_adjustment
                    min_bound, max_bound = self.scale_bounds.get(key, (0.01, 1.0))
                    self.scales[key] = max(min_bound, min(max_bound, new_scale))

    def get_scales(self) -> Dict[str, float]:
        return self.scales


class SparseDictionaryTrainer:
    def __init__(self, model_path: str, layer_dims, device: str = "cuda", beta: float = 0.98):
        self.device = device

        # Load model with memory optimizations enabled
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)

        if torch.cuda.is_available():
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.hidden_size = self.model.config.hidden_size

        if hasattr(self.model, "generation_config"):
            current_temp = getattr(self.model.generation_config, "temperature", 0.6)
            self.model.generation_config.temperature = max(0.5, min(current_temp, 0.7))
        else:
            self.model.generation_config = type("GenerationConfig", (object,), {})()
            self.model.generation_config.temperature = 0.6

        # Create SAEs with proper device placement
        self.saes = nn.ModuleList([
            ManifoldAwareAutoencoder(self.hidden_size, hidden_dim).to(self.device)
            for _, hidden_dim in layer_dims
        ])

        self.beta = beta
        self.difficulty_history = []
        self.best_metrics = {}
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        wandb.init(project="sparse_dictionary_learning",
                   config={"layers": layer_dims, "model": model_path, "beta": beta, "model_type": "manifold_aware"})

        # Add CUDA memory management for better performance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.set_per_process_memory_fraction(0.95)
            except:
                pass

        # Storage for precomputed hidden states
        self.train_hiddens = None
        self.val_hiddens = None

    def save_checkpoint(self, layer_idx: int, epoch: int, optimizer, scaler) -> None:
        path = os.path.join(self.checkpoint_dir, f"layer_{layer_idx}_epoch_{epoch}.pt")
        sae = self.saes[layer_idx]
        checkpoint_data = {
            "model_state": sae.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "metrics": self.best_metrics,
            "feature_ema": sae.feature_ema,
            "semantic_cache": sae.semantic_cache,
            "model_type": "manifold_aware"
        }
        torch.save(checkpoint_data, path)

    def log_metrics(self, layer_idx: int, epoch: int, train_loss: float, val_metrics: FeatureMetrics,
                    optimizer) -> None:
        log_data = {
            f"layer_{layer_idx}/train_loss": train_loss,
            f"layer_{layer_idx}/lr": optimizer.param_groups[0]["lr"]
        }
        log_data.update({f"layer_{layer_idx}/val_{k}": v for k, v in vars(val_metrics).items()})
        if "gate_activity" in self.best_metrics:
            log_data[f"layer_{layer_idx}/gate_activity"] = self.best_metrics["gate_activity"]
        wandb.log(log_data)

        print(f"\nLayer {layer_idx} Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print("Validation Metrics:")
        for key, value in vars(val_metrics).items():
            print(f"{key:>20}: {value:.4f}")
        if "gate_activity" in self.best_metrics:
            print(f"{'gate_activity':>20}: {self.best_metrics['gate_activity']:.4f}")

    def prepare_dataset(self, dataset_name="wikitext", subset="wikitext-103-v1", max_samples=5000):
        # Use streaming to avoid loading entire dataset
        dataset = load_dataset(dataset_name, subset, streaming=True)

        # Calculate exactly how many samples we need for optimization
        required_samples = max_samples
        print(f"Loading exactly {required_samples} samples from the dataset...")

        # Take exactly the number we need using list + take
        train_samples = list(dataset['train'].take(required_samples))

        from datasets import Dataset
        small_dataset = {
            'train': Dataset.from_dict({k: [ex[k] for ex in train_samples] for k in train_samples[0].keys()}),
        }

        # Same for validation
        if 'validation' in dataset:
            val_required = required_samples // 5  # 20% of training for validation
            val_samples = list(dataset['validation'].take(val_required))
            small_dataset['validation'] = Dataset.from_dict(
                {k: [ex[k] for ex in val_samples] for k in val_samples[0].keys()})

        print(f"Dataset loaded: {len(small_dataset['train'])} training samples")

        # Clear memory
        del train_samples
        if 'validation' in dataset:
            del val_samples
        gc.collect()

        return small_dataset

    def create_dataloaders(self, dataset, batch_size=32, val_split=0.1, max_length=512, num_workers=0,
                           max_train_batches=100, max_val_batches=20):
        import platform
        is_windows = platform.system() == 'Windows'

        # On Windows, ensure we use 0 workers to avoid issues
        if is_windows:
            num_workers = 0

        train_samples_needed = min(max_train_batches * batch_size if max_train_batches > 0 else len(dataset["train"]),
                                   len(dataset["train"]))
        val_samples_needed = max_val_batches * batch_size if max_val_batches > 0 else None

        # First subsample to exact number needed BEFORE tokenization to save memory
        if train_samples_needed < len(dataset["train"]):
            import random
            train_indices = random.sample(range(len(dataset["train"])), train_samples_needed)
            reduced_train = dataset["train"].select(train_indices)
        else:
            reduced_train = dataset["train"]

        print(f"Tokenizing exactly {len(reduced_train)} training examples (reduced from {len(dataset['train'])})")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"] if "text" in examples else examples,
                padding="max_length",  # Always use max_length padding for uniform tensors
                truncation=True,
                max_length=max_length,  # Use fixed length instead of calculating from text
                return_tensors="pt"
            )

        # Custom collate function that properly handles padding
        def safe_collate(batch):
            if isinstance(batch[0], dict):
                result = {}
                for key in batch[0].keys():
                    if torch.is_tensor(batch[0][key]):
                        # Check if all tensors have the same shape
                        shapes = [example[key].shape for example in batch]
                        if all(shape == shapes[0] for shape in shapes):
                            # If same shape, we can stack
                            result[key] = torch.stack([example[key] for example in batch])
                        else:
                            # Different shapes - pad to max length in this batch
                            if key == 'input_ids' or key == 'attention_mask':
                                # Find max length in this dimension
                                max_len = max(example[key].size(0) for example in batch)
                                # Pad each tensor to max_len
                                padded_tensors = []
                                for example in batch:
                                    tensor = example[key]
                                    if tensor.size(0) < max_len:
                                        padding = torch.zeros(max_len - tensor.size(0), dtype=tensor.dtype,
                                                              device=tensor.device)
                                        tensor = torch.cat([tensor, padding])
                                    padded_tensors.append(tensor)
                                result[key] = torch.stack(padded_tensors)
                            else:
                                # For other keys, just convert to list
                                result[key] = [example[key] for example in batch]
                    else:
                        result[key] = [example[key] for example in batch]
                return result
            else:
                # Handle non-dictionary batches
                if torch.is_tensor(batch[0]):
                    shapes = [x.shape for x in batch]
                    if all(shape == shapes[0] for shape in shapes):
                        return torch.stack(batch)
                    else:
                        # Different shapes - can't stack
                        return batch
                else:
                    return batch

        tokenize_batch_size = 32  # Smaller batch size for tokenization to avoid OOM
        tokenized_train = reduced_train.map(
            tokenize_function,
            batched=True,
            batch_size=tokenize_batch_size,
            remove_columns=["text"] if "text" in reduced_train.column_names else None
        )
        tokenized_train.set_format("torch")

        if "validation" not in dataset:
            # Create validation split only from the already reduced training set
            train_size = int((1 - val_split) * len(tokenized_train))
            val_size = len(tokenized_train) - train_size
            train_dataset, val_dataset = random_split(tokenized_train, [train_size, val_size])
        else:
            train_dataset = tokenized_train

            # Handle validation set similarly
            if val_samples_needed and val_samples_needed < len(dataset["validation"]):
                val_indices = random.sample(range(len(dataset["validation"])), val_samples_needed)
                reduced_val = dataset["validation"].select(val_indices)
            else:
                reduced_val = dataset["validation"]

            print(f"Tokenizing exactly {len(reduced_val)} validation examples")

            tokenized_val = reduced_val.map(
                tokenize_function,
                batched=True,
                batch_size=tokenize_batch_size,
                remove_columns=["text"] if "text" in dataset["validation"].column_names else None
            )
            tokenized_val.set_format("torch")
            val_dataset = tokenized_val

        # DataLoader configuration adjusted for Windows compatibility
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
            persistent_workers=False,  # Disable persistent workers everywhere to be safe
            collate_fn=safe_collate
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
            persistent_workers=False,  # Disable persistent workers everywhere to be safe
            collate_fn=safe_collate
        )

        print(
            f"Created optimized dataloaders: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches")
        print(f"Using {num_workers} workers, persistent_workers=False")

        return train_dataloader, val_dataloader

    def use_dummy_data(self, train_batches=100, val_batches=20, batch_size=32, seq_length=512):
        device = self.device if self.device != "cpu" else None
        train_data = [torch.randint(0, 10000, (batch_size, seq_length), device=device) for _ in range(train_batches)]
        val_data = [torch.randint(0, 10000, (batch_size, seq_length), device=device) for _ in range(val_batches)]
        print(f"Created dummy data: {train_batches} training batches, {val_batches} validation batches")
        return train_data, val_data

    def precompute_hidden_states(self, data_loader, is_train=True):
        """Precompute hidden states for all layers to optimize training"""
        print(f"Precomputing {'training' if is_train else 'validation'} hidden states for all layers...")

        # Create storage for each layer
        layer_hiddens = {i: [] for i in range(len(self.saes))}

        self.model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Precomputing hidden states")):
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                # Get all hidden states in a single forward pass
                outputs = self.model(
                    input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False
                )

                # Store each layer's hidden states
                for layer_idx in range(len(self.saes)):
                    hidden = outputs.hidden_states[layer_idx].detach()
                    # Reshape to (batch_size * seq_len, hidden_size)
                    reshaped = hidden.view(-1, self.hidden_size)
                    # Store in half precision to save memory
                    layer_hiddens[layer_idx].append(reshaped.to(torch.float16))

                # Clear cache every few batches
                if batch_idx % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Concatenate all batches for each layer
        consolidated = {}
        for layer_idx, hiddens in layer_hiddens.items():
            consolidated[layer_idx] = hiddens

        # Store in the appropriate attribute
        if is_train:
            self.train_hiddens = consolidated
        else:
            self.val_hiddens = consolidated

        print(f"Finished precomputing hidden states for {len(consolidated)} layers")

        # Clear memory
        torch.cuda.empty_cache()

    def train_layer_precomputed(self, layer_idx: int, train_hiddens, val_hiddens, epochs: int = 50,
                                batch_size: int = 64):
        sae = self.saes[layer_idx]
        optimizer = optim.AdamW(sae.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.999))
        scaler = GradScaler()
        loss_scaler = DynamicLossScaler({
            "reconstruction": 1.0,
            "sparsity": 0.1,
            "diversity": 0.05,
            "stability": 0.025,
            "orthogonality": 0.01,
            "nonlinearity": 0.1,
            "gate_diversity": 0.05
        })

        # Determine total steps for scheduler
        total_batches = sum(len(batches) for batches in train_hiddens)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=epochs,
            steps_per_epoch=total_batches, pct_start=0.3,
            anneal_strategy='cos'
        )

        # Use moderate gradient accumulation to reduce updates but not too much
        grad_accum_steps = max(2, 64 // batch_size) if layer_idx == 0 else max(4, 128 // batch_size)

        # Compute metrics every N batches - moderate frequency
        metrics_compute_interval = 10 if layer_idx == 0 else 20
        orthogonal_project_interval = 15 if layer_idx == 0 else 30

        # Configure mixed precision settings
        compute_dtype = torch.float32
        mixed_dtype = torch.float16 if self.device == "cuda" else torch.bfloat16

        for epoch in range(epochs):
            sae.train()
            epoch_metrics = defaultdict(float)
            optimizer.zero_grad(set_to_none=True)

            # Force garbage collection between epochs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Create one progress bar for all batches
            progress_bar = tqdm(enumerate(train_hiddens), desc=f"Layer {layer_idx} Epoch {epoch}")
            sae.sparsity_controller.update_target(epoch, epochs)

            global_batch_idx = 0

            for batch_idx, hidden_batch in progress_bar:
                # Move hidden states to device
                hidden = hidden_batch.to(self.device, non_blocking=True)
                global_batch_idx += 1

                # Determine if we should compute full metrics
                compute_metrics = global_batch_idx % metrics_compute_interval == 0

                # Ensure we use the scaler and autocast consistently
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                              dtype=mixed_dtype, enabled=True):

                    # Forward pass
                    recon, activations, gates = sae(hidden, batch_idx=global_batch_idx)
                    recon_loss = F.mse_loss(recon, hidden)
                    sparsity = (activations.abs() < 0.01).float().mean()
                    sparsity_penalty = sae.sparsity_controller.compute_penalty(sparsity.item())

                    if compute_metrics:
                        # Compute expensive metrics only periodically
                        frame_potential = sae.encoder.compute_frame_potential()
                        coherence, diversity, stability, nonlinearity, _ = sae.compute_feature_statistics(activations)
                        ortho_loss = sae.decoder.compute_frame_potential()
                        gate_diversity_loss = sae.compute_gate_diversity_loss(gates)

                        # Store computed metrics to reuse
                        sae._last_frame_potential = frame_potential.detach()
                        sae._last_coherence = coherence
                        sae._last_diversity = diversity
                        sae._last_stability = stability
                        sae._last_nonlinearity = nonlinearity
                        sae._last_ortho_loss = ortho_loss.detach()
                        sae._last_gate_div = gate_diversity_loss.detach()
                    else:
                        # Reuse previously computed metrics
                        frame_potential = getattr(sae, '_last_frame_potential', torch.tensor(0.1, device=self.device))
                        coherence = getattr(sae, '_last_coherence', 0.2)
                        diversity = getattr(sae, '_last_diversity', 0.7)
                        stability = getattr(sae, '_last_stability', 0.8)
                        nonlinearity = getattr(sae, '_last_nonlinearity', 0.5)
                        ortho_loss = getattr(sae, '_last_ortho_loss', torch.tensor(0.05, device=self.device))
                        gate_diversity_loss = getattr(sae, '_last_gate_div', torch.tensor(0.2, device=self.device))

                    scales = loss_scaler.get_scales()

                    # Calculate loss with all components
                    loss = (
                                   scales["reconstruction"] * recon_loss +
                                   scales["sparsity"] * sparsity_penalty * frame_potential +
                                   scales["diversity"] * (1 - diversity) +
                                   scales["stability"] * (1 - stability) +
                                   scales["orthogonality"] * ortho_loss +
                                   scales["gate_diversity"] * gate_diversity_loss +
                                   scales.get("nonlinearity", 0.1) * (1 - nonlinearity)
                           ) / grad_accum_steps

                    metrics = {
                        "loss": loss.item() * grad_accum_steps,
                        "sparsity": sparsity.item(),
                        "diversity": diversity,
                        "stability": stability,
                        "recon": recon_loss.item(),
                        "ortho": ortho_loss.item() if hasattr(ortho_loss, 'item') else 0.0,
                        "gate_div": gate_diversity_loss.item() if hasattr(gate_diversity_loss, 'item') else 0.0,
                        "nonlinearity": nonlinearity
                    }

                # Scale and backward
                scaler.scale(loss).backward()

                # Only update occasionally
                if (global_batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    # Project orthogonally periodically but not every step
                    if global_batch_idx % orthogonal_project_interval == 0:
                        sae.encoder.project_to_stiefel()
                        sae.decoder.project_to_stiefel()

                # Update display
                for k, v in metrics.items():
                    epoch_metrics[k] += v

                progress_bar.set_postfix({k: v / (batch_idx + 1) for k, v in epoch_metrics.items()})

                # Explicitly delete tensors
                del recon, activations, gates, hidden, recon_loss, loss

                # Force memory cleanup periodically
                if global_batch_idx % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # End of epoch processing
            avg_metrics = {k: v / len(train_hiddens) for k, v in epoch_metrics.items()}
            loss_scaler.update_scales(avg_metrics)

            # Validate every 5 epochs or at the end
            do_validation = (epoch % 5 == 0) or (epoch == epochs - 1)
            if do_validation:
                val_metrics = self.validate_precomputed(layer_idx, val_hiddens)
                self.save_checkpoint(layer_idx, epoch, optimizer, scaler)
                self.log_metrics(layer_idx, epoch, avg_metrics["loss"], val_metrics, optimizer)
            # Save checkpoint at end of epoch without validation
            elif epoch % 10 == 0:
                self.save_checkpoint(layer_idx, epoch, optimizer, scaler)

            # Clear memory between epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def validate_precomputed(self, layer_idx: int, val_hiddens) -> FeatureMetrics:
        sae = self.saes[layer_idx]
        sae.eval()
        total = FeatureMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        gate_activity = 0.0

        with torch.no_grad():
            for batch_idx, hidden_batch in enumerate(val_hiddens):
                hidden = hidden_batch.to(self.device)
                recon, activations, gates = sae(hidden, batch_idx=batch_idx)
                recon_loss = F.mse_loss(recon, hidden)
                sparsity = (activations.abs() < 0.01).float().mean()
                frame_potential = sae.encoder.compute_frame_potential()
                coherence, diversity, stability, nonlinearity, _ = sae.compute_feature_statistics(activations)
                semantic = sae.track_semantic_consistency(activations, "val")
                ortho_loss = sae.decoder.compute_frame_potential()
                gate_activity += (1.0 - gates.mean()).item()

                batch_metrics = FeatureMetrics(
                    float(sparsity.item()),
                    float(coherence),
                    float(stability),
                    float(diversity),
                    float(semantic),
                    float(recon_loss.item()),
                    float(ortho_loss.item()),
                    float(nonlinearity)
                )

                for field in FeatureMetrics.__dataclass_fields__:
                    setattr(total, field, getattr(total, field) + getattr(batch_metrics, field))

        num_batches = len(val_hiddens)
        for field in FeatureMetrics.__dataclass_fields__:
            setattr(total, field, getattr(total, field) / num_batches)

        self.best_metrics["gate_activity"] = gate_activity / num_batches
        return total

    def train_layer(self, layer_idx: int, train_data, val_data, epochs: int = 50, batch_size: int = 64):
        """Traditional training loop that computes hidden states on the fly"""
        sae = self.saes[layer_idx]
        optimizer = optim.AdamW(sae.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.999))
        scaler = GradScaler()
        loss_scaler = DynamicLossScaler({
            "reconstruction": 1.0,
            "sparsity": 0.1,
            "diversity": 0.05,
            "stability": 0.025,
            "orthogonality": 0.01,
            "nonlinearity": 0.1,
            "gate_diversity": 0.05
        })
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs,
                                                        steps_per_epoch=len(train_data), pct_start=0.3,
                                                        anneal_strategy='cos')

        # Define consistent dtype for computation to avoid mixing float16/float32
        compute_dtype = torch.float32

        # Set autocast dtype - use float16 on CUDA, bfloat16 otherwise
        mixed_dtype = torch.float16 if self.device == "cuda" else torch.bfloat16

        # Use moderate gradient accumulation to reduce updates
        grad_accum_steps = max(2, 64 // batch_size) if layer_idx == 0 else max(4, 128 // batch_size)

        # Compute metrics every N batches - moderate frequency
        metrics_compute_interval = 10 if layer_idx == 0 else 20
        orthogonal_project_interval = 15 if layer_idx == 0 else 30

        for epoch in range(epochs):
            sae.train()
            epoch_metrics = defaultdict(float)
            optimizer.zero_grad(set_to_none=True)

            # Force garbage collection between epochs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            progress_bar = tqdm(train_data, desc=f"Layer {layer_idx} Epoch {epoch}")
            sae.sparsity_controller.update_target(epoch, epochs)

            for batch_idx, batch in enumerate(progress_bar):
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                else:
                    input_ids = batch.to(self.device, non_blocking=True)

                # Extract hidden states with consistent dtype
                with torch.no_grad():
                    outputs = self.model(
                        input_ids,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False
                    )
                    # Convert hidden states to compute_dtype for consistency
                    hidden = outputs.hidden_states[layer_idx].view(-1, self.hidden_size).to(compute_dtype)
                    del outputs

                # Determine if we should compute full metrics
                compute_metrics = batch_idx % metrics_compute_interval == 0

                # Use autocast for mixed precision
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                              dtype=mixed_dtype, enabled=True):

                    # Forward pass through SAE
                    recon, activations, gates = sae(hidden, batch_idx=batch_idx)
                    recon_loss = F.mse_loss(recon, hidden)
                    sparsity = (activations.abs() < 0.01).float().mean()
                    sparsity_penalty = sae.sparsity_controller.compute_penalty(sparsity.item())

                    if compute_metrics:
                        # Compute expensive metrics only periodically
                        frame_potential = sae.encoder.compute_frame_potential()
                        coherence, diversity, stability, nonlinearity, _ = sae.compute_feature_statistics(activations)
                        ortho_loss = sae.decoder.compute_frame_potential()
                        gate_diversity_loss = sae.compute_gate_diversity_loss(gates)

                        # Store computed metrics to reuse
                        sae._last_frame_potential = frame_potential.detach()
                        sae._last_coherence = coherence
                        sae._last_diversity = diversity
                        sae._last_stability = stability
                        sae._last_nonlinearity = nonlinearity
                        sae._last_ortho_loss = ortho_loss.detach()
                        sae._last_gate_div = gate_diversity_loss.detach()
                    else:
                        # Reuse previously computed metrics
                        frame_potential = getattr(sae, '_last_frame_potential', torch.tensor(0.1, device=self.device))
                        coherence = getattr(sae, '_last_coherence', 0.2)
                        diversity = getattr(sae, '_last_diversity', 0.7)
                        stability = getattr(sae, '_last_stability', 0.8)
                        nonlinearity = getattr(sae, '_last_nonlinearity', 0.5)
                        ortho_loss = getattr(sae, '_last_ortho_loss', torch.tensor(0.05, device=self.device))
                        gate_diversity_loss = getattr(sae, '_last_gate_div', torch.tensor(0.2, device=self.device))

                    scales = loss_scaler.get_scales()

                    # Calculate loss with all components
                    loss = (
                                   scales["reconstruction"] * recon_loss +
                                   scales["sparsity"] * sparsity_penalty * frame_potential +
                                   scales["diversity"] * (1 - diversity) +
                                   scales["stability"] * (1 - stability) +
                                   scales["orthogonality"] * ortho_loss +
                                   scales["gate_diversity"] * gate_diversity_loss +
                                   scales.get("nonlinearity", 0.1) * (1 - nonlinearity)
                           ) / grad_accum_steps

                    metrics = {
                        "loss": loss.item() * grad_accum_steps,
                        "sparsity": sparsity.item(),
                        "diversity": diversity,
                        "stability": stability,
                        "recon": recon_loss.item(),
                        "ortho": ortho_loss.item() if hasattr(ortho_loss, 'item') else 0.0,
                        "gate_div": gate_diversity_loss.item() if hasattr(gate_diversity_loss, 'item') else 0.0,
                        "nonlinearity": nonlinearity
                    }

                # Scale and backward
                scaler.scale(loss).backward()

                # Only update occasionally
                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    # Project orthogonally periodically but not every step
                    if batch_idx % orthogonal_project_interval == 0:
                        sae.encoder.project_to_stiefel()
                        sae.decoder.project_to_stiefel()

                # Update display
                for k, v in metrics.items():
                    epoch_metrics[k] += v

                progress_bar.set_postfix({k: v / (batch_idx + 1) for k, v in epoch_metrics.items()})

                # Explicitly delete tensors
                del recon, activations, gates, hidden, recon_loss, loss

                # Force memory cleanup periodically
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # End of epoch processing
            avg_metrics = {k: v / len(train_data) for k, v in epoch_metrics.items()}
            loss_scaler.update_scales(avg_metrics)

            # Validate every 5 epochs or at the end
            do_validation = (epoch % 5 == 0) or (epoch == epochs - 1)
            if do_validation:
                val_metrics = self.validate(layer_idx, val_data)
                self.save_checkpoint(layer_idx, epoch, optimizer, scaler)
                self.log_metrics(layer_idx, epoch, avg_metrics["loss"], val_metrics, optimizer)
            # Save checkpoint at end of epoch without validation
            elif epoch % 10 == 0:
                self.save_checkpoint(layer_idx, epoch, optimizer, scaler)

            # Clear memory between epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def validate(self, layer_idx: int, val_data) -> FeatureMetrics:
        sae = self.saes[layer_idx]
        sae.eval()
        total = FeatureMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        gate_activity = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data):
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                else:
                    input_ids = batch.to(self.device)
                hidden = self.model(input_ids, output_hidden_states=True).hidden_states[layer_idx]
                hidden = hidden.view(-1, self.hidden_size).float()
                recon, activations, gates = sae(hidden, batch_idx=batch_idx)
                recon_loss = F.mse_loss(recon, hidden)
                sparsity = (activations.abs() < 0.01).float().mean()
                frame_potential = sae.encoder.compute_frame_potential()
                coherence, diversity, stability, nonlinearity, _ = sae.compute_feature_statistics(activations)
                semantic = sae.track_semantic_consistency(activations, "val")
                ortho_loss = sae.decoder.compute_frame_potential()
                gate_activity += (1.0 - gates.mean()).item()
                batch_metrics = FeatureMetrics(
                    float(sparsity.item()),
                    float(coherence),
                    float(stability),
                    float(diversity),
                    float(semantic),
                    float(recon_loss.item()),
                    float(ortho_loss.item()),
                    float(nonlinearity)
                )
                for field in FeatureMetrics.__dataclass_fields__:
                    setattr(total, field, getattr(total, field) + getattr(batch_metrics, field))
        num_batches = len(val_data)
        for field in FeatureMetrics.__dataclass_fields__:
            setattr(total, field, getattr(total, field) / num_batches)
        self.best_metrics["gate_activity"] = gate_activity / num_batches
        return total

    def train_all(self, train_data, val_data, epochs: int = 50, batch_size: int = 64, use_precomputation: bool = True):
        """Train all layers with the option to use precomputation for performance"""
        if use_precomputation:
            # Precompute hidden states for all layers (major optimization)
            self.precompute_hidden_states(train_data, is_train=True)
            self.precompute_hidden_states(val_data, is_train=False)

            # Free up GPU memory by moving model to CPU after precomputation
            self.model = self.model.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Train each layer using the precomputed hidden states
            for layer_idx in range(len(self.saes)):
                print(f"\n=== Training Layer {layer_idx} with Precomputed Hidden States ===\n")
                train_hiddens = self.train_hiddens[layer_idx]
                val_hiddens = self.val_hiddens[layer_idx]
                self.train_layer_precomputed(layer_idx, train_hiddens, val_hiddens, epochs, batch_size)

                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()

            # Clean up precomputed states after training
            self.train_hiddens = None
            self.val_hiddens = None
            gc.collect()
        else:
            # Traditional training loop (computationally expensive)
            for layer_idx in range(len(self.saes)):
                print(f"\n=== Training Layer {layer_idx} ===\n")
                self.train_layer(layer_idx, train_data, val_data, epochs, batch_size)
                torch.cuda.empty_cache()
                gc.collect()


if __name__ == "__main__":
    import argparse
    import platform

    parser = argparse.ArgumentParser(description="Train sparse dictionary learning model")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-103-v1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_batches", type=int, default=100)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--layer_dims", type=str, default="2048,4096")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--dummy_data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimize_memory", action="store_true", help="Apply aggressive memory optimizations")
    parser.add_argument("--eval_interval", type=int, default=5, help="Run validation every N epochs")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoints every N epochs")
    parser.add_argument("--precompute", action="store_true", help="Use hidden state precomputation for speed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Windows-specific settings - disable multiprocessing completely
    is_windows = platform.system() == 'Windows'
    num_workers = 0  # Just use 0 workers to avoid all multiprocessing issues

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        if args.optimize_memory:
            try:
                import os

                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'
            except:
                pass

            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            except:
                pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda" and args.optimize_memory:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Memory: {gpu_mem:.2f} GB")

        layer_dims = [(None, int(dim)) for dim in args.layer_dims.split(",")]
        max_dict_size = max([dim for _, dim in layer_dims])

        if "1.5B" in args.model or "1.8B" in args.model:
            base_batch_size = 32 if max_dict_size <= 4096 else 24
        elif "7B" in args.model:
            base_batch_size = 16 if max_dict_size <= 4096 else 8
        else:
            base_batch_size = 24 if max_dict_size <= 4096 else 16

        args.batch_size = args.batch_size or base_batch_size
        print(f"Using optimized batch size: {args.batch_size}")

    layer_dims = [(None, int(dim)) for dim in args.layer_dims.split(",")]
    trainer = SparseDictionaryTrainer(model_path=args.model, layer_dims=layer_dims, device=device)
    trainer.checkpoint_dir = args.checkpoint_dir

    print(f"Model: {args.model}")
    print(f"Dictionary sizes: {[dim for _, dim in layer_dims]}")

    if args.dummy_data:
        train_dataloader, val_dataloader = trainer.use_dummy_data(
            train_batches=args.train_batches,
            val_batches=args.val_batches,
            batch_size=args.batch_size,
            seq_length=args.max_length
        )
    else:
        # Use smaller batch size for memory efficiency
        if args.train_batches <= 8:
            effective_max_samples = args.batch_size * args.train_batches * 2
            print(f"Using minimal dataset size: {effective_max_samples} samples")
        else:
            # Never use more than 2000 samples
            effective_max_samples = min(2000, args.max_samples)
            print(f"Limiting to {effective_max_samples} samples for memory efficiency")

        # Adjust batch size based on layer
        if args.batch_size > 16:
            print(f"Reducing batch size from {args.batch_size} to 16 for memory efficiency")
            args.batch_size = 16  # Force smaller batch size

        dataset = trainer.prepare_dataset(
            dataset_name=args.dataset,
            subset=args.subset,
            max_samples=effective_max_samples
        )

        train_dataloader, val_dataloader = trainer.create_dataloaders(
            dataset,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_train_batches=args.train_batches,
            max_val_batches=args.val_batches,
            num_workers=0  # Completely disable workers
        )

    try:
        trainer.train_all(
            train_dataloader,
            val_dataloader,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_precomputation=args.precompute
        )
        print("Training complete!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()