import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.linalg import svd, qr
from torch.nn.utils.parametrizations import orthogonal
import wandb
import os
from tqdm import tqdm
from dataclasses import dataclass


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
    def __init__(self, target_sparsity=0.2, adaptation_rate=0.05, tolerance=0.1):
        self.target_sparsity = target_sparsity
        self.adaptation_rate = adaptation_rate
        self.tolerance = tolerance
        self.history = []
        self.moving_avg = None
        self.beta = 0.9

    def compute_penalty(self, current_sparsity):
        # Update moving average for stability
        if self.moving_avg is None:
            self.moving_avg = current_sparsity
        else:
            self.moving_avg = self.beta * self.moving_avg + (1 - self.beta) * current_sparsity

        # Compute error with respect to moving average
        error = self.target_sparsity - self.moving_avg
        self.history.append(error)

        # Asymmetric penalty function
        if error > 0:
            return torch.exp(torch.tensor(error / self.tolerance))
        else:
            return 0.5 * torch.log1p(torch.tensor(abs(error) / self.tolerance))

    def update_target(self, epoch, total_epochs):
        # Progressive sparsity schedule
        progress = epoch / total_epochs
        base_sparsity = 0.2
        max_sparsity = 0.5
        self.target_sparsity = base_sparsity + (max_sparsity - base_sparsity) * \
                               (1 - np.exp(-5 * progress))


class StiefelGrassmannianDictionary(nn.Module):
    def __init__(self, input_dim, dict_size, eps=1e-6, tau=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.tau = tau
        self.eps = eps
        self.dictionary = orthogonal(nn.Linear(input_dim, dict_size, bias=False))
        self.initialise_stiefel()

    def initialise_stiefel(self):
        with torch.no_grad():
            if self.input_dim >= self.dict_size:
                random_matrix = torch.randn(self.input_dim, self.dict_size)
                Q, _ = qr(random_matrix)
                self.dictionary.weight.data = Q.t()
            else:
                random_matrix = torch.randn(self.dict_size, self.input_dim)
                Q, _ = qr(random_matrix)
                self.dictionary.weight.data = Q[:self.input_dim].t()

    def project_to_stiefel(self):
        with torch.no_grad():
            W = self.dictionary.weight
            U, _, Vh = svd(W.t(), full_matrices=False)
            self.dictionary.weight.data = Vh.t() @ U.t()

    def compute_frame_potential(self):
        W = self.dictionary.weight
        gram = W @ W.t() if self.input_dim >= self.dict_size else W.t() @ W
        identity = torch.eye(gram.size(0), device=gram.device)
        return torch.norm(gram - identity) ** 2

    def forward(self, x):
        return self.dictionary(x)


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, coherence_penalty=0.1, diversity_weight=0.1, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.encoder = StiefelGrassmannianDictionary(input_dim, hidden_dim, eps=self.eps)
        self.decoder = StiefelGrassmannianDictionary(hidden_dim, input_dim, eps=self.eps)
        self.activation = nn.ReLU()
        self.coherence_penalty = coherence_penalty
        self.diversity_weight = diversity_weight
        self.sparsity_controller = AdaptiveSparsityController()
        self.feature_history = []
        self.semantic_cache = {}
        # New attributes for stability tracking
        self.feature_ema = None
        self.ema_decay = 0.99

    def compute_feature_statistics(self, activations):
        with torch.autocast(device_type='cuda', enabled=False):
            act = activations.float()
            batch_size = act.size(0)

            # Normalisation with improved numerical stability
            act_norm = act / (torch.norm(act, dim=1, keepdim=True) + self.eps)

            # Coherence computation
            gram = torch.mm(act_norm, act_norm.t())
            mask = torch.triu(torch.ones_like(gram), diagonal=1).bool()
            coherence = torch.abs(gram[mask]).mean()

            # Enhanced diversity computation using eigenvalue dispersion
            cov_matrix = (act.T @ act) / (batch_size - 1)
            eigenvals = torch.linalg.eigvalsh(cov_matrix + self.eps * torch.eye(cov_matrix.size(0),
                                                                                device=cov_matrix.device))
            normalised_eigenvals = eigenvals / (torch.sum(eigenvals) + self.eps)
            diversity = 1.0 - (torch.max(normalised_eigenvals) - torch.min(normalised_eigenvals)) / \
                        (1.0 + torch.std(normalised_eigenvals))

            # Stability computation with EMA
            if self.feature_ema is None:
                self.feature_ema = act.mean(dim=0)
            else:
                current_mean = act.mean(dim=0)
                self.feature_ema = self.ema_decay * self.feature_ema + \
                                   (1 - self.ema_decay) * current_mean

            mean_deviation = torch.norm(act.mean(dim=0) - self.feature_ema)
            stability = torch.exp(-mean_deviation / self.eps)

            # Ensure metrics are properly bounded
            coherence = torch.clamp(coherence, 0, 1)
            diversity = torch.clamp(diversity, 0, 1)
            stability = torch.clamp(stability, 0, 1)

            return coherence.item(), diversity.item(), stability.item()

    def compute_feature_diversity(self, activations):
        coherence, diversity, _ = self.compute_feature_statistics(activations)
        return diversity

    def track_semantic_consistency(self, activations, batch_id):
        with torch.autocast(device_type='cuda', enabled=False):
            act = activations.float()

            if batch_id in self.semantic_cache:
                prev_act = self.semantic_cache[batch_id].float()

                act_norm = act / (torch.norm(act, dim=1, keepdim=True) + self.eps)
                prev_norm = prev_act / (torch.norm(prev_act, dim=1, keepdim=True) + self.eps)

                consistency = torch.mean(torch.sum(act_norm * prev_norm, dim=1))

                self.semantic_cache[batch_id] = 0.9 * prev_act + 0.1 * act.detach()

                return consistency.item()

            self.semantic_cache[batch_id] = act.detach()
            return 1.0

    def forward(self, x):
        x = x.to(torch.float32)

        # Input normalisation with improved numerical stability
        x_centred = x - x.mean(dim=1, keepdim=True)
        x_scaled = x_centred / (torch.norm(x_centred, dim=1, keepdim=True) + self.eps)

        # Encoder pass with gradient scaling
        encoded = self.encoder(x_scaled)

        # Unified sparsity mechanism
        abs_encoded = torch.abs(encoded).float()
        threshold = torch.quantile(abs_encoded, 0.8, dim=1, keepdim=True)
        threshold = threshold.detach()

        if self.training:
            # Soft thresholding with temperature scaling
            temp = 0.1
            soft_threshold = torch.sigmoid((abs_encoded - threshold) / temp)
            activations = encoded * soft_threshold

            # Add minimal noise to prevent dead features
            if activations.abs().mean() < self.eps:
                noise_scale = 0.01 * threshold.mean()
                activations = activations + torch.randn_like(activations) * noise_scale
        else:
            # Hard thresholding for inference
            activations = encoded * (abs_encoded > threshold).float()

        # Decoder pass with normalisation preservation
        decoded = self.decoder(activations)

        # Scale recovery with proper statistics preservation
        output = decoded * torch.norm(x_centred, dim=1, keepdim=True) + x.mean(dim=1, keepdim=True)

        return output, activations


class AdaptiveCurriculumTrainer:
    def __init__(self, model_path, layer_dims, device='cuda', beta=0.98):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.hidden_size = self.model.config.hidden_size
        self.saes = nn.ModuleList([SparseAutoencoder(self.hidden_size, hid_dim).to(device)
                                   for layer_idx, hid_dim in enumerate(dim[1] for dim in layer_dims)])
        self.beta = beta
        self.difficulty_history = []
        self.best_metrics = {}
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        wandb.init(project="sdl_steifel", config={
            "layers": layer_dims,
            "model": model_path,
            "beta": beta
        })

    def save_checkpoint(self, layer_idx, epoch):
        path = f'{self.checkpoint_dir}/layer_{layer_idx}_epoch_{epoch}.pt'
        torch.save({
            'model': self.saes[layer_idx].state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'metrics': self.best_metrics
        }, path)

    def load_checkpoint(self, layer_idx, epoch):
        path = f'{self.checkpoint_dir}/layer_{layer_idx}_epoch_{epoch}.pt'
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.saes[layer_idx].load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.best_metrics = checkpoint['metrics']
            return True
        return False

    def compute_batch_difficulty(self, loss):
        if not self.difficulty_history:
            self.difficulty_history.append(loss)
            return loss
        new_diff = self.beta * self.difficulty_history[-1] + (1 - self.beta) * loss
        self.difficulty_history.append(new_diff)
        return (new_diff - min(self.difficulty_history)) / (
                max(self.difficulty_history) - min(self.difficulty_history) + 1e-6)

    def train_layer(self, layer_idx, train_data, val_data, epochs=100, batch_size=32):
        sae = self.saes[layer_idx]
        self.optimizer = optim.AdamW(sae.parameters(), lr=5e-4, weight_decay=1e-5)
        self.scaler = GradScaler()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)

        # Define loss scaling factors
        loss_scales = {
            'reconstruction': 1.0,
            'sparsity': 0.1,
            'diversity': 0.05,
            'stability': 0.025,
            'orthogonality': 0.01
        }

        for epoch in range(epochs):
            sae.train()
            epoch_loss = 0
            progress = tqdm(train_data, desc=f"Layer {layer_idx} Epoch {epoch}")

            sae.sparsity_controller.update_target(epoch, epochs)

            for batch in progress:
                x = batch.to(self.device)
                with torch.no_grad():
                    hidden_states = self.model(x, output_hidden_states=True).hidden_states[layer_idx]
                    hidden_states = hidden_states.view(-1, self.hidden_size).float()

                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    recon, activations = sae(hidden_states)

                    # Compute all loss components
                    recon_loss = nn.MSELoss()(recon, hidden_states)
                    sparsity = (activations.abs() < 0.01).float().mean()
                    sparsity_penalty = sae.sparsity_controller.compute_penalty(sparsity.item())
                    frame_potential = sae.encoder.compute_frame_potential()
                    coherence, diversity, stability = sae.compute_feature_statistics(activations)
                    ortho_loss = sae.decoder.compute_frame_potential()

                    # Combine losses with scaling factors
                    loss = loss_scales['reconstruction'] * recon_loss + \
                           loss_scales['sparsity'] * sparsity_penalty * frame_potential + \
                           loss_scales['diversity'] * (1 - diversity) + \
                           loss_scales['stability'] * (1 - stability) + \
                           loss_scales['orthogonality'] * ortho_loss

                # Gradient scaling and clipping
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(sae.parameters(), 0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Project weights onto Stiefel manifold
                sae.encoder.project_to_stiefel()
                sae.decoder.project_to_stiefel()

                epoch_loss += loss.item()
                progress.set_postfix({
                    'loss': loss.item(),
                    'sparsity': sparsity.item(),
                    'diversity': diversity,
                    'stability': stability
                })

            avg_loss = epoch_loss / len(train_data)
            scheduler.step(avg_loss)
            val_metrics = self.validate(layer_idx, val_data)

            self.save_checkpoint(layer_idx, epoch)
            self.log_metrics(layer_idx, epoch, avg_loss, val_metrics)

    def validate(self, layer_idx, val_data):
        sae = self.saes[layer_idx]
        sae.eval()
        total_metrics = FeatureMetrics(0, 0, 0, 0, 0, 0, 0)

        with torch.no_grad():
            for batch in val_data:
                x = batch.to(self.device)
                hidden_states = self.model(x, output_hidden_states=True).hidden_states[layer_idx]
                hidden_states = hidden_states.view(-1, self.hidden_size).float()
                recon, activations = sae(hidden_states)

                # Compute primary metrics
                recon_loss = nn.MSELoss()(recon, hidden_states)
                sparsity = (activations.abs() < 0.01).float().mean()
                frame_potential = sae.encoder.compute_frame_potential()

                # Get feature statistics
                coherence, diversity, stability = sae.compute_feature_statistics(activations)

                # Compute semantic consistency
                semantic = sae.track_semantic_consistency(activations, 'val')

                # Compute orthogonality loss
                ortho_loss = sae.decoder.compute_frame_potential()

                # Ensure all values are Python floats before creating FeatureMetrics
                metrics = FeatureMetrics(
                    sparsity=float(sparsity.item()),
                    coherence=float(coherence),
                    stability=float(stability),
                    diversity=float(diversity),
                    semantic_consistency=float(semantic),
                    recon_error=float(recon_loss.item()),
                    ortho_loss=float(ortho_loss.item())
                )

                # Accumulate metrics
                for field in FeatureMetrics.__dataclass_fields__:
                    current_value = getattr(total_metrics, field)
                    new_value = getattr(metrics, field)
                    setattr(total_metrics, field, current_value + new_value)

        # Average the accumulated metrics
        batch_count = len(val_data)
        for field in FeatureMetrics.__dataclass_fields__:
            current_value = getattr(total_metrics, field)
            setattr(total_metrics, field, current_value / batch_count)

        return total_metrics

    def log_metrics(self, layer_idx, epoch, train_loss, val_metrics):
        log_data = {
            f'layer_{layer_idx}/train_loss': train_loss,
            f'layer_{layer_idx}/lr': self.optimizer.param_groups[0]['lr'],
            **{f'layer_{layer_idx}/val_{k}': v for k, v in vars(val_metrics).items()}
        }
        wandb.log(log_data)

        print(f"\nLayer {layer_idx} Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print("Validation Metrics:")
        for k, v in vars(val_metrics).items():
            print(f"{k:>20}: {v:.4f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = AdaptiveCurriculumTrainer(
        model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        layer_dims=[(None, 2048), (None, 4096)],
        device=device
    )

    print("Model config:", trainer.model.config)  # Access through trainer instance
    print("Hidden size:", trainer.model.config.hidden_size)

    train_data = [torch.randint(0, 10000, (32, 512)).to(device) for _ in range(100)]
    val_data = [torch.randint(0, 10000, (16, 512)).to(device) for _ in range(20)]

    for layer_idx in range(len(trainer.saes)):
        trainer.train_layer(layer_idx, train_data, val_data, epochs=50)