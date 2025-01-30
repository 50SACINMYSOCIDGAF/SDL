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
    def __init__(self, target_sparsity=0.1, adaptation_rate=0.01, tolerance=0.05):
        self.target_sparsity = target_sparsity
        self.adaptation_rate = adaptation_rate
        self.tolerance = tolerance
        self.history = []

    def compute_penalty(self, current_sparsity):
        error = self.target_sparsity - current_sparsity
        self.history.append(error)
        return torch.exp(torch.tensor(error / self.tolerance)) if error > 0 else torch.log1p(
            torch.tensor(abs(error) / self.tolerance))

    def update_target(self, epoch, total_epochs):
        progress = epoch / total_epochs
        self.target_sparsity = 0.05 + 0.15 * (1 - np.exp(-3 * progress))


class StiefelGrassmannianDictionary(nn.Module):
    def __init__(self, input_dim, dict_size, tau=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.tau = tau
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
        return torch.norm(gram - torch.eye(gram.size(0), device=gram.device)) ** 2

    def forward(self, x):
        return self.dictionary(x)


class EnhancedSparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, coherence_penalty=0.1, diversity_weight=0.1):
        super().__init__()
        self.encoder = StiefelGrassmannianDictionary(input_dim, hidden_dim)
        self.decoder = StiefelGrassmannianDictionary(hidden_dim, input_dim)
        self.activation = nn.ReLU()
        self.coherence_penalty = coherence_penalty
        self.diversity_weight = diversity_weight
        self.sparsity_controller = AdaptiveSparsityController()
        self.feature_history = []
        self.semantic_cache = {}

    def compute_feature_diversity(self, activations):
        with torch.autocast(device_type='cuda', enabled=False):
            activations_float = activations.float()
            corr = torch.corrcoef(activations_float.T)
            eigenvalues = torch.linalg.eigvalsh(corr)
            return torch.var(eigenvalues / eigenvalues.sum())

    def track_semantic_consistency(self, activations, batch_id):
        with torch.autocast(device_type='cuda', enabled=False):
            activations_float = activations.float()
            if batch_id in self.semantic_cache:
                prev = self.semantic_cache[batch_id].float()
                cos_sim = nn.CosineSimilarity(dim=1)(activations_float, prev)
                self.semantic_cache[batch_id] = activations_float.detach()
                return cos_sim.mean().item()
            self.semantic_cache[batch_id] = activations_float.detach()
            return 1.0

    def forward(self, x):
        activations = self.activation(self.encoder(x))
        return self.decoder(activations), activations


class AdaptiveCurriculumTrainer:
    def __init__(self, model_path, layer_dims, device='cuda', beta=0.98):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.hidden_size = self.model.config.hidden_size
        self.saes = nn.ModuleList([EnhancedSparseAutoencoder(self.hidden_size, hid_dim).to(device)
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
        self.optimizer = optim.AdamW(sae.parameters(), lr=1e-3, weight_decay=0.01)
        self.scaler = GradScaler()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)

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
                with autocast():
                    recon, activations = sae(hidden_states)
                    recon_loss = nn.MSELoss()(recon, hidden_states)
                    sparsity = (activations == 0).float().mean()
                    sparsity_penalty = sae.sparsity_controller.compute_penalty(sparsity.item())
                    frame_potential = sae.encoder.compute_frame_potential()
                    diversity_loss = sae.compute_feature_diversity(activations)
                    ortho_loss = sae.decoder.compute_frame_potential()

                    loss = recon_loss + sparsity_penalty * frame_potential + \
                           sae.diversity_weight * diversity_loss + 0.1 * ortho_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                sae.encoder.project_to_stiefel()
                sae.decoder.project_to_stiefel()

                epoch_loss += loss.item()
                progress.set_postfix({'loss': loss.item(), 'sparsity': sparsity.item()})

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

                recon_loss = nn.MSELoss()(recon, hidden_states)
                sparsity = (activations == 0).float().mean()
                frame_potential = sae.encoder.compute_frame_potential()
                diversity = sae.compute_feature_diversity(activations)
                ortho_loss = sae.decoder.compute_frame_potential()
                stability = torch.std(activations, dim=0).mean()
                semantic = sae.track_semantic_consistency(activations, 'val')

                metrics = FeatureMetrics(
                    sparsity.item(), frame_potential.item(), stability.item(),
                    diversity.item(), semantic, recon_loss.item(), ortho_loss.item()
                )

                for field in FeatureMetrics.__dataclass_fields__:
                    setattr(total_metrics, field, getattr(total_metrics, field) + getattr(metrics, field))

        for field in FeatureMetrics.__dataclass_fields__:
            setattr(total_metrics, field, getattr(total_metrics, field) / len(val_data))

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

    train_data = [torch.randint(0, 10000, (32, 512)).to(device) for _ in range(100)]
    val_data = [torch.randint(0, 10000, (16, 512)).to(device) for _ in range(20)]

    for layer_idx in range(len(trainer.saes)):
        trainer.train_layer(layer_idx, train_data, val_data, epochs=50)