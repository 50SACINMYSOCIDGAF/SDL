import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.linalg import matrix_norm, svd, qr
from typing import Dict, List, Tuple, Optional
import wandb
from dataclasses import dataclass
from torch.nn.utils.parametrizations import orthogonal


@dataclass
class FeatureMetrics:
    """Structured container for feature analysis metrics"""
    sparsity: float
    coherence: float
    stability: float
    diversity: float
    semantic_consistency: float


class AdaptiveSparsityController:
    """Controls sparsity through dynamic regulation"""

    def __init__(self,
                 target_sparsity: float = 0.1,
                 adaptation_rate: float = 0.01,
                 tolerance: float = 0.05):
        self.target_sparsity = target_sparsity
        self.adaptation_rate = adaptation_rate
        self.tolerance = tolerance
        self.history = []

    def compute_penalty(self, current_sparsity: float) -> float:
        """Compute adaptive sparsity penalty based on current state"""
        error = self.target_sparsity - current_sparsity
        self.history.append(error)

        # Exponential scaling for better gradient properties
        if error > 0:
            return np.exp(error / self.tolerance)
        return np.log1p(abs(error) / self.tolerance)

    def update_target(self, epoch: int, total_epochs: int):
        """Update target sparsity based on training progress"""
        progress = epoch / total_epochs
        self.target_sparsity = 0.05 + 0.15 * (1 - np.exp(-3 * progress))


class StiefelGrassmannianDictionary(nn.Module):
    """Dictionary learning with Stiefel manifold constraints"""

    def __init__(self,
                 input_dim: int,
                 dict_size: int,
                 tau: float = 0.01):  # Retraction parameter
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.tau = tau

        # Initialize on Stiefel manifold
        self.dictionary = orthogonal(nn.Linear(input_dim, dict_size))
        self.initialise_stiefel()

    def initialise_stiefel(self):
        """Initialise dictionary on Stiefel manifold using QR decomposition"""
        with torch.no_grad():
            Q, R = qr(torch.randn(self.input_dim, self.dict_size))
            self.dictionary.weight.copy_(Q.t())

    def project_to_stiefel(self):
        """Project onto Stiefel manifold using polar decomposition"""
        with torch.no_grad():
            W = self.dictionary.weight
            WtW = torch.matmul(W.t(), W)
            U, S, V = svd(WtW)
            Q = torch.matmul(U, V.t())
            self.dictionary.weight.copy_(torch.matmul(W, Q))

    def compute_frame_potential(self) -> torch.Tensor:
        """Compute frame potential with respect to Stiefel geometry"""
        W = self.dictionary.weight
        W_norm = W / torch.norm(W, dim=1, keepdim=True)
        gram = torch.mm(W_norm, W_norm.t())
        frame_potential = torch.sum(gram ** 2) - torch.sum(torch.diagonal(gram) ** 2)
        return frame_potential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Stiefel constraints"""
        return self.dictionary(x)


class EnhancedSparseAutoencoder(nn.Module):
    """Advanced sparse autoencoder with Stiefel manifold constraints"""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 coherence_penalty: float = 0.1,
                 diversity_weight: float = 0.1):
        super().__init__()
        self.encoder = StiefelGrassmannianDictionary(input_dim, hidden_dim)
        self.decoder = StiefelGrassmannianDictionary(hidden_dim, input_dim)
        self.activation = nn.ReLU()

        # Hyperparameters
        self.coherence_penalty = coherence_penalty
        self.diversity_weight = diversity_weight

        # Sparsity control
        self.sparsity_controller = AdaptiveSparsityController()

        # Feature tracking
        self.feature_history = []
        self.semantic_cache = {}

    def compute_feature_diversity(self, activations: torch.Tensor) -> torch.Tensor:
        """Enhanced diversity measurement using eigenvalue analysis"""
        feature_correlations = torch.corrcoef(activations.T)

        # Compute eigenvalue distribution
        eigenvalues = torch.linalg.eigvalsh(feature_correlations)

        # Measure deviation from uniform distribution
        uniform = torch.ones_like(eigenvalues) / len(eigenvalues)
        diversity_penalty = torch.norm(eigenvalues - uniform)

        return diversity_penalty

    def track_semantic_consistency(self,
                                   activations: torch.Tensor,
                                   batch_id: str) -> float:
        """Track semantic consistency with temporal averaging"""
        if batch_id in self.semantic_cache:
            prev_activations = self.semantic_cache[batch_id]

            # Compute cosine similarity for semantic consistency
            norm_current = torch.norm(activations, dim=1, keepdim=True)
            norm_prev = torch.norm(prev_activations, dim=1, keepdim=True)

            cos_sim = torch.sum(activations * prev_activations, dim=1) / (norm_current * norm_prev)
            consistency = torch.mean(cos_sim).item()

            self.semantic_cache[batch_id] = activations.detach()
            return consistency

        self.semantic_cache[batch_id] = activations.detach()
        return 1.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with enhanced feature extraction"""
        activations = self.activation(self.encoder(x))
        reconstructed = self.decoder(activations)
        return reconstructed, activations


class AdaptiveCurriculumTrainer:
    """Enhanced trainer with curriculum learning and Stiefel optimisation"""

    def __init__(self,
                 model_path: str,
                 layer_dims: List[Tuple[int, int]],
                 device: str = 'cuda',
                 beta: float = 0.98):
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.saes = nn.ModuleList([
            EnhancedSparseAutoencoder(in_dim, hid_dim).to(device)
            for in_dim, hid_dim in layer_dims
        ])

        self.device = device
        self.beta = beta
        self.difficulty_history = []

        wandb.init(project="advanced_sdl", config={
            "layer_dims": layer_dims,
            "model": model_path,
            "beta": beta
        })

    def compute_batch_difficulty(self, loss: float) -> float:
        """Enhanced difficulty estimation using EMA"""
        if not self.difficulty_history:
            self.difficulty_history.append(loss)
            return loss

        avg_difficulty = self.difficulty_history[-1]
        new_difficulty = self.beta * avg_difficulty + (1 - self.beta) * loss
        self.difficulty_history.append(new_difficulty)

        # Normalise difficulty score
        return (new_difficulty - min(self.difficulty_history)) / \
            (max(self.difficulty_history) - min(self.difficulty_history) + 1e-6)

    def train_layer(self,
                    layer_idx: int,
                    train_data: torch.Tensor,
                    epochs: int = 100,
                    batch_size: int = 32):
        sae = self.saes[layer_idx]
        optimizer = optim.AdamW(sae.parameters(), lr=1e-3)
        scaler = GradScaler()

        for epoch in range(epochs):
            total_metrics = FeatureMetrics(0., 0., 0., 0., 0.)
            batch_count = 0

            # Update sparsity targets
            sae.sparsity_controller.update_target(epoch, epochs)

            for batch_idx, batch in enumerate(train_data):
                x = batch.to(self.device)

                with torch.no_grad():
                    outputs = self.model(x, output_hidden_states=True)
                    x = outputs.hidden_states[layer_idx]

                optimizer.zero_grad()

                with autocast():
                    recon_x, activations = sae(x)

                    # Enhanced loss computation
                    recon_loss = nn.MSELoss()(recon_x, x)
                    frame_potential = sae.encoder.compute_frame_potential()
                    diversity_loss = sae.compute_feature_diversity(activations)

                    # Adaptive sparsity
                    current_sparsity = (activations == 0).float().mean().item()
                    sparsity_penalty = sae.sparsity_controller.compute_penalty(current_sparsity)

                    # Total loss with adaptive weights
                    difficulty = self.compute_batch_difficulty(recon_loss.item())

                    loss = (recon_loss +
                            sparsity_penalty * frame_potential +
                            sae.diversity_weight * diversity_loss)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                # Project onto Stiefel manifold
                sae.encoder.project_to_stiefel()
                sae.decoder.project_to_stiefel()

                # Compute comprehensive metrics
                with torch.no_grad():
                    metrics = FeatureMetrics(
                        sparsity=current_sparsity,
                        coherence=frame_potential.item(),
                        stability=torch.std(activations, dim=0).mean().item(),
                        diversity=diversity_loss.item(),
                        semantic_consistency=sae.track_semantic_consistency(
                            activations, f"batch_{batch_idx}")
                    )

                    total_metrics = FeatureMetrics(
                        *(getattr(total_metrics, field) + getattr(metrics, field)
                          for field in FeatureMetrics.__dataclass_fields__)
                    )
                    batch_count += 1

            # Average metrics
            avg_metrics = FeatureMetrics(
                *(getattr(total_metrics, field) / batch_count
                  for field in FeatureMetrics.__dataclass_fields__)
            )

            # Enhanced logging
            wandb.log({
                f"layer_{layer_idx}/loss": loss.item(),
                f"layer_{layer_idx}/difficulty": difficulty,
                f"layer_{layer_idx}/sparsity_target": sae.sparsity_controller.target_sparsity,
                **{f"layer_{layer_idx}/{field}": getattr(avg_metrics, field)
                   for field in FeatureMetrics.__dataclass_fields__}
            })

            print(f"Epoch {epoch} Metrics:")
            for field in FeatureMetrics.__dataclass_fields__:
                print(f"- {field}: {getattr(avg_metrics, field):.4f}")


if __name__ == "__main__":
    # Configuration
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    layer_dims = [(768, 2048), (2048, 4096)]

    # Initialize trainer
    trainer = AdaptiveCurriculumTrainer(model_path, layer_dims)

    # Generate sample data
    train_data = [torch.randint(0, 10000, (32, 512)) for _ in range(100)]

    # Train progressively
    for layer_idx in range(len(layer_dims)):
        print(f"\nTraining Layer {layer_idx}")
        trainer.train_layer(layer_idx, train_data)