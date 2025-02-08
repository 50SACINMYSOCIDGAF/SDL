import os
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.nn.utils.parametrizations import orthogonal
from torch.linalg import svd, qr

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

    def compute_penalty(self, current_sparsity: float) -> torch.Tensor:
        if self.moving_avg is None:
            self.moving_avg = current_sparsity
        else:
            self.moving_avg = self.beta * self.moving_avg + (1 - self.beta) * current_sparsity

        error = self.target_sparsity - self.moving_avg
        self.history.append(error)
        if error > 0:
            return torch.exp(torch.tensor(error / self.tolerance))
        else:
            return 0.5 * torch.log1p(torch.tensor(abs(error) / self.tolerance))

    def update_target(self, epoch: int, total_epochs: int) -> None:
        progress = epoch / total_epochs
        base_sparsity = 0.2
        max_sparsity = 0.5
        self.target_sparsity = base_sparsity + (max_sparsity - base_sparsity) * (1 - np.exp(-5 * progress))


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
        self.feature_ema = None  # For stability tracking using EMA
        self.ema_decay = 0.99

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
            diversity = 1.0 - (torch.max(normalized_eigenvals) - torch.min(normalized_eigenvals)) / (
                        1.0 + torch.std(normalized_eigenvals))

            current_mean = act.mean(dim=0)
            if self.feature_ema is None:
                self.feature_ema = current_mean
                stability = torch.tensor(1.0, device=current_mean.device)
            else:
                delta = current_mean - self.feature_ema
                relative_change = torch.norm(delta) / (torch.norm(self.feature_ema) + self.eps)

                adaptive_rate = torch.exp(-relative_change / self.eps)
                self.feature_ema = (1 - adaptive_rate) * self.feature_ema + adaptive_rate * current_mean

                stability = torch.exp(-relative_change)

            coherence = torch.clamp(coherence, 0, 1)
            diversity = torch.clamp(diversity, 0, 1)
            stability = torch.clamp(stability, 0.1, 1)  # Minimum stability of 0.1

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
        threshold = torch.quantile(abs_encoded, 0.8, dim=1, keepdim=True).detach()
        if self.training:
            soft_temp = 0.1  # temperature for soft thresholding
            soft_threshold = torch.sigmoid((abs_encoded - threshold) / soft_temp)
            activations = encoded * soft_threshold
            if activations.abs().mean() < self.eps:
                noise_scale = 0.01 * threshold.mean()
                activations = activations + torch.randn_like(activations) * noise_scale
        else:
            activations = encoded * (abs_encoded > threshold).float()
        decoded = self.decoder(activations)
        output = decoded * torch.norm(x_centered, dim=1, keepdim=True) + x.mean(dim=1, keepdim=True)
        return output, activations


class AdaptiveCurriculumTrainer:
    def __init__(self, model_path: str, layer_dims, device: str = "cuda", beta: float = 0.98):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.hidden_size = self.model.config.hidden_size

        # Clamp generation temperature between 0.5 and 0.7.
        if hasattr(self.model, "generation_config"):
            current_temp = getattr(self.model.generation_config, "temperature", 0.6)
            self.model.generation_config.temperature = max(0.5, min(current_temp, 0.7))
        else:
            self.model.generation_config = type("GenerationConfig", (object,), {})()
            self.model.generation_config.temperature = 0.6

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
            "beta": beta
        })

    def save_checkpoint(self, layer_idx: int, epoch: int, optimizer, scaler) -> None:
        path = os.path.join(self.checkpoint_dir, f"layer_{layer_idx}_epoch_{epoch}.pt")
        torch.save({
            "model_state": self.saes[layer_idx].state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "metrics": self.best_metrics,
            "feature_ema": self.saes[layer_idx].feature_ema,  # Add this
            "semantic_cache": self.saes[layer_idx].semantic_cache  # Add this
        }, path)

    def load_checkpoint(self, layer_idx: int, epoch: int, optimizer, scaler) -> bool:
        path = os.path.join(self.checkpoint_dir, f"layer_{layer_idx}_epoch_{epoch}.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.saes[layer_idx].load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            self.best_metrics = checkpoint["metrics"]

            # Safely handle new checkpoint fields
            if "feature_ema" in checkpoint:
                self.saes[layer_idx].feature_ema = checkpoint["feature_ema"]
            if "semantic_cache" in checkpoint:
                self.saes[layer_idx].semantic_cache = checkpoint["semantic_cache"]
            return True
        return False

    def compute_batch_difficulty(self, loss: float) -> float:
        if not self.difficulty_history:
            self.difficulty_history.append(loss)
            return loss
        new_diff = self.beta * self.difficulty_history[-1] + (1 - self.beta) * loss
        self.difficulty_history.append(new_diff)
        diff_range = max(self.difficulty_history) - min(self.difficulty_history) + 1e-6
        return (new_diff - min(self.difficulty_history)) / diff_range

    def train_layer(self, layer_idx: int, train_data, val_data, epochs: int = 100, batch_size: int = 32,
                    start_epoch: int = 0):
        sae = self.saes[layer_idx]
        optimizer = optim.AdamW(sae.parameters(), lr=5e-4, weight_decay=1e-5)
        scaler = GradScaler()

        # Load checkpoint if starting from later epoch
        if start_epoch > 0:
            print(f"Loading checkpoint from Layer {layer_idx} Epoch {start_epoch - 1}")
            self.load_checkpoint(layer_idx, start_epoch - 1, optimizer, scaler)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        loss_scales = {
            "reconstruction": 1.0,
            "sparsity": 0.1,
            "diversity": 0.05,
            "stability": 0.025,
            "orthogonality": 0.01
        }
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16

        for epoch in range(start_epoch, epochs):
            sae.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_data, desc=f"Layer {layer_idx} Epoch {epoch}")
            sae.sparsity_controller.update_target(epoch, epochs)

            for batch in progress_bar:
                x = batch.to(self.device)
                with torch.no_grad():
                    hidden = self.model(x, output_hidden_states=True).hidden_states[layer_idx]
                    hidden = hidden.view(-1, self.hidden_size).float()
                optimizer.zero_grad()
                with torch.autocast(device_type=self.device, dtype=dtype):
                    recon, activations = sae(hidden)
                    recon_loss = nn.MSELoss()(recon, hidden)
                    sparsity = (activations.abs() < 0.01).float().mean()
                    sparsity_penalty = sae.sparsity_controller.compute_penalty(sparsity.item())
                    frame_potential = sae.encoder.compute_frame_potential()
                    coherence, diversity, stability = sae.compute_feature_statistics(activations)
                    ortho_loss = sae.decoder.compute_frame_potential()
                    loss = (loss_scales["reconstruction"] * recon_loss +
                            loss_scales["sparsity"] * sparsity_penalty * frame_potential +
                            loss_scales["diversity"] * (1 - diversity) +
                            loss_scales["stability"] * (1 - stability) +
                            loss_scales["orthogonality"] * ortho_loss)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(sae.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()

                sae.encoder.project_to_stiefel()
                sae.decoder.project_to_stiefel()

                epoch_loss += loss.item()
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "sparsity": sparsity.item(),
                    "diversity": diversity,
                    "stability": stability
                })

            avg_loss = epoch_loss / len(train_data)
            scheduler.step(avg_loss)
            val_metrics = self.validate(layer_idx, val_data)
            self.save_checkpoint(layer_idx, epoch, optimizer, scaler)
            self.log_metrics(layer_idx, epoch, avg_loss, val_metrics, optimizer)

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
                batch_metrics = FeatureMetrics(
                    sparsity=float(sparsity.item()),
                    coherence=float(coherence),
                    stability=float(stability),
                    diversity=float(diversity),
                    semantic_consistency=float(semantic),
                    recon_error=float(recon_loss.item()),
                    ortho_loss=float(ortho_loss.item())
                )
                for field in FeatureMetrics.__dataclass_fields__:
                    setattr(total, field, getattr(total, field) + getattr(batch_metrics, field))
        num_batches = len(val_data)
        for field in FeatureMetrics.__dataclass_fields__:
            setattr(total, field, getattr(total, field) / num_batches)
        return total

    def log_metrics(self, layer_idx: int, epoch: int, train_loss: float, val_metrics: FeatureMetrics, optimizer) -> None:
        log_data = {
            f"layer_{layer_idx}/train_loss": train_loss,
            f"layer_{layer_idx}/lr": optimizer.param_groups[0]["lr"],
        }
        log_data.update({f"layer_{layer_idx}/val_{k}": v for k, v in vars(val_metrics).items()})
        wandb.log(log_data)
        print(f"\nLayer {layer_idx} Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print("Validation Metrics:")
        for key, value in vars(val_metrics).items():
            print(f"{key:>20}: {value:.4f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = AdaptiveCurriculumTrainer(
        model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        layer_dims=[(None, 2048), (None, 4096)],
        device=device
    )
    print("Model config:", trainer.model.config)
    print("Hidden size:", trainer.model.config.hidden_size)

    train_data = [torch.randint(0, 10000, (32, 512)).to(device) for _ in range(100)]
    val_data = [torch.randint(0, 10000, (16, 512)).to(device) for _ in range(20)]

    # Single layer training - no loop needed
    trainer.train_layer(1, train_data, val_data, epochs=50, start_epoch=30)