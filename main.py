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
    def __init__(self, target_sparsity: float = 0.2, adaptation_rate: float = 0.05, tolerance: float = 0.1):
        self.target_sparsity = target_sparsity
        self.adaptation_rate = adaptation_rate
        self.tolerance = tolerance
        self.history = []
        self.moving_avg = None
        self.beta = 0.9
        self.min_penalty = 0.01
        self.max_penalty = 10.0
    def compute_penalty(self, current_sparsity: float) -> torch.Tensor:
        if self.moving_avg is None:
            self.moving_avg = current_sparsity
        else:
            self.moving_avg = self.beta * self.moving_avg + (1 - self.beta) * current_sparsity
        error = self.target_sparsity - self.moving_avg
        self.history.append(error)
        if error > 0:
            penalty = torch.exp(torch.tensor(error / self.tolerance))
        else:
            penalty = 0.5 * torch.log1p(torch.tensor(abs(error) / self.tolerance))
        return torch.clamp(penalty, self.min_penalty, self.max_penalty)
    def update_target(self, epoch: int, total_epochs: int) -> None:
        progress = epoch / total_epochs
        base_sparsity = 0.15
        max_sparsity = 0.4
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
    def reinitialize_unstable_features(self, activations: torch.Tensor, stability_threshold: float = 0.2):
        with torch.no_grad():
            feature_stabilities = []
            for i in range(activations.size(1)):
                if self.feature_ema is not None:
                    feat_ema = self.feature_ema[i]
                    feat_current = activations[:, i].mean()
                    relative_change = abs(feat_current - feat_ema) / (abs(feat_ema) + self.eps)
                    stability = 1.0 / (1.0 + relative_change)
                    feature_stabilities.append(stability)
                    if stability < stability_threshold:
                        rand_vec = torch.randn(self.encoder.input_dim, device=activations.device)
                        rand_vec = rand_vec / torch.norm(rand_vec)
                        self.encoder.dictionary.weight.data[i] = rand_vec
                        if i < self.decoder.input_dim:
                            self.decoder.dictionary.weight.data[:, i] = rand_vec
    def compute_feature_statistics(self, activations: torch.Tensor):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            act = activations.float()
            batch_size = act.size(0)
            act_norm = act / (torch.norm(act, dim=1, keepdim=True) + self.eps)
            gram = torch.mm(act_norm, act_norm.t())
            mask = torch.triu(torch.ones_like(gram), diagonal=1).bool()
            coherence = torch.abs(gram[mask]).mean()
            cov_matrix = (act.T @ act) / (batch_size - 1)
            eye = torch.eye(cov_matrix.size(0), device=cov_matrix.device)
            eigenvals = torch.linalg.eigvalsh(cov_matrix + self.eps * eye)
            normalized_eigenvals = eigenvals / (torch.sum(eigenvals) + self.eps)
            diversity = 1.0 - torch.max(normalized_eigenvals) / (torch.sum(normalized_eigenvals) + self.eps)
            if self.feature_ema is None:
                self.feature_ema = act.mean(dim=0)
                stability = torch.tensor(1.0, device=act.device)
            else:
                current_mean = act.mean(dim=0)
                delta = current_mean - self.feature_ema
                relative_change = torch.norm(delta) / (torch.norm(self.feature_ema) + self.eps)
                adaptive_rate = torch.clip(0.1 * torch.exp(-5 * relative_change), 0.001, 0.1)
                self.feature_ema = (1 - adaptive_rate) * self.feature_ema + adaptive_rate * current_mean
                stability = 1.0 / (1.0 + relative_change)
            coherence = torch.clamp(coherence, 0, 1)
            diversity = torch.clamp(diversity, 0, 1)
            stability = torch.clamp(stability, 0.15, 1)
            return coherence.item(), diversity.item(), stability.item()
    def track_semantic_consistency(self, activations: torch.Tensor, batch_id: str):
        act_norm = activations / (torch.norm(activations, dim=1, keepdim=True) + self.eps)
        if batch_id in self.semantic_cache:
            prev_norm = self.semantic_cache[batch_id]
            consistency = torch.mean(torch.sum(act_norm * prev_norm, dim=1))
            self.semantic_cache[batch_id] = 0.9 * prev_norm + 0.1 * act_norm.detach()
            return consistency.item()
        self.semantic_cache[batch_id] = act_norm.detach()
        return 1.0
    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)
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
            if activations.abs().mean() < self.eps:
                noise_scale = 0.01 * threshold.mean()
                activations = activations + torch.randn_like(activations) * noise_scale
            self.reinitialize_unstable_features(activations)
        else:
            activations = encoded * (abs_encoded > torch.quantile(abs_encoded, 0.8, dim=1, keepdim=True)).float()
        decoded = self.decoder(activations)
        output = decoded * torch.norm(x_centered, dim=1, keepdim=True) + x.mean(dim=1, keepdim=True)
        return output, activations

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
    def __init__(self, model_path: str, layer_dims, device: str = "cuda", beta: float = 0.98):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.hidden_size = self.model.config.hidden_size
        if hasattr(self.model, "generation_config"):
            current_temp = getattr(self.model.generation_config, "temperature", 0.6)
            self.model.generation_config.temperature = max(0.5, min(current_temp, 0.7))
        else:
            self.model.generation_config = type("GenerationConfig", (object,), {})()
            self.model.generation_config.temperature = 0.6
        self.saes = nn.ModuleList([SparseAutoencoder(self.hidden_size, hidden_dim).to(self.device) for _, hidden_dim in layer_dims])
        self.beta = beta
        self.difficulty_history = []
        self.best_metrics = {}
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        wandb.init(project="sdl_steifel", config={"layers": layer_dims, "model": model_path, "beta": beta})
    def save_checkpoint(self, layer_idx: int, epoch: int, optimizer, scaler) -> None:
        path = os.path.join(self.checkpoint_dir, f"layer_{layer_idx}_epoch_{epoch}.pt")
        torch.save({"model_state": self.saes[layer_idx].state_dict(), "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(), "metrics": self.best_metrics, "feature_ema": self.saes[layer_idx].feature_ema, "semantic_cache": self.saes[layer_idx].semantic_cache}, path)
    def train_layer(self, layer_idx: int, train_data, val_data, epochs: int = 50, batch_size: int = 64):
        sae = self.saes[layer_idx]
        optimizer = optim.AdamW(sae.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.999))
        scaler = GradScaler()
        loss_scaler = DynamicLossScaler({"reconstruction": 1.0, "sparsity": 0.1, "diversity": 0.05, "stability": 0.025, "orthogonality": 0.01})
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_data), pct_start=0.3, anneal_strategy='cos')
        grad_accum_steps = max(1, 256 // batch_size)
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        for epoch in range(epochs):
            sae.train()
            epoch_metrics = defaultdict(float)
            optimizer.zero_grad(set_to_none=True)
            progress_bar = tqdm(train_data, desc=f"Layer {layer_idx} Epoch {epoch}")
            sae.sparsity_controller.update_target(epoch, epochs)
            for batch_idx, batch in enumerate(progress_bar):
                x = batch.to(self.device, non_blocking=True)
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                    hidden = self.model(x, output_hidden_states=True).hidden_states[layer_idx]
                    hidden = hidden.view(-1, self.hidden_size)
                with torch.cuda.amp.autocast(dtype=dtype):
                    recon, activations = sae(hidden)
                    recon_loss = F.mse_loss(recon, hidden)
                    sparsity = (activations.abs() < 0.01).float().mean()
                    sparsity_penalty = sae.sparsity_controller.compute_penalty(sparsity.item())
                    frame_potential = sae.encoder.compute_frame_potential()
                    coherence, diversity, stability = sae.compute_feature_statistics(activations)
                    ortho_loss = sae.decoder.compute_frame_potential()
                    scales = loss_scaler.get_scales()
                    loss = (scales["reconstruction"] * recon_loss + scales["sparsity"] * sparsity_penalty * frame_potential + scales["diversity"] * (1 - diversity) + scales["stability"] * (1 - stability) + scales["orthogonality"] * ortho_loss) / grad_accum_steps
                scaler.scale(loss).backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    sae.encoder.project_to_stiefel()
                    sae.decoder.project_to_stiefel()
                metrics = {"loss": loss.item() * grad_accum_steps, "sparsity": sparsity.item(), "diversity": diversity, "stability": stability, "recon": recon_loss.item(), "ortho": ortho_loss.item()}
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
                batch_metrics = FeatureMetrics(float(sparsity.item()), float(coherence), float(stability), float(diversity), float(semantic), float(recon_loss.item()), float(ortho_loss.item()))
                for field in FeatureMetrics.__dataclass_fields__:
                    setattr(total, field, getattr(total, field) + getattr(batch_metrics, field))
        num_batches = len(val_data)
        for field in FeatureMetrics.__dataclass_fields__:
            setattr(total, field, getattr(total, field) / num_batches)
        return total
    def log_metrics(self, layer_idx: int, epoch: int, train_loss: float, val_metrics: FeatureMetrics, optimizer) -> None:
        log_data = {f"layer_{layer_idx}/train_loss": train_loss, f"layer_{layer_idx}/lr": optimizer.param_groups[0]["lr"]}
        log_data.update({f"layer_{layer_idx}/val_{k}": v for k, v in vars(val_metrics).items()})
        wandb.log(log_data)
        print(f"\nLayer {layer_idx} Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print("Validation Metrics:")
        for key, value in vars(val_metrics).items():
            print(f"{key:>20}: {value:.4f}")
    def train_all(self, train_data, val_data, epochs: int = 50, batch_size: int = 64):
        for layer_idx in range(len(self.saes)):
            self.train_layer(layer_idx, train_data, val_data, epochs, batch_size)

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = AdaptiveCurriculumTrainer(model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", layer_dims=[(None, 2048), (None, 4096)], device=device)
    print("Model config:", trainer.model.config)
    print("Hidden size:", trainer.model.config.hidden_size)
    train_data = [torch.randint(0, 10000, (32, 512)).to(device) for _ in range(100)]
    val_data = [torch.randint(0, 10000, (16, 512)).to(device) for _ in range(20)]
    trainer.train_all(train_data, val_data, epochs=50)