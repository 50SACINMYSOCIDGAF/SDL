import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.parametrizations import orthogonal
from contextlib import nullcontext
import gc
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np


def check_cuda_availability():
    print("CUDA Environment Check:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")


class StiefelGrassmannianDictionary(nn.Module):
    def __init__(self, input_dim, dict_size, eps=1e-6, tau=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.tau = tau
        self.eps = eps
        self.dictionary = orthogonal(nn.Linear(input_dim, dict_size, bias=False))

    def forward(self, x):
        return self.dictionary(x)


class EnhancedSparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.encoder = StiefelGrassmannianDictionary(input_dim, hidden_dim, eps=self.eps)
        self.decoder = StiefelGrassmannianDictionary(hidden_dim, input_dim, eps=self.eps)

    def forward(self, x):
        # Preprocess input: center and scale
        x = x.float()
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_scaled = x_centered / (torch.norm(x_centered, dim=1, keepdim=True) + self.eps)

        model_dtype = self.encoder.dictionary.weight.dtype
        x_scaled = x_scaled.to(dtype=model_dtype)

        # Encoder pass
        encoded = self.encoder(x_scaled)

        # Sparsity: only keep top 20% activations (by absolute value)
        abs_encoded = encoded.float()
        threshold = torch.quantile(abs_encoded.abs(), 0.8, dim=1, keepdim=True)
        mask = (abs_encoded.abs() > threshold).to(dtype=model_dtype)
        activations = encoded * mask

        # Decoder pass to reconstruct
        decoded = self.decoder(activations)
        decoded = decoded.float()
        output = decoded * torch.norm(x_centered, dim=1, keepdim=True) + x.mean(dim=1, keepdim=True)

        return output, activations


@dataclass
class DictionaryAnalysis:
    atom_id: int
    activation_frequency: float
    orthogonality_measure: float
    coherence_score: float
    top_correlations: dict


class SemanticFeatureAnalyzer:
    """
    Loads the language model and SDL.
    Provides methods for semantic analysis, such as clustering dictionary atoms
    and assessing the causal impact of individual atoms on a given text.
    """

    def __init__(self,
                 sdl_path: str = "layer_1_epoch_49.pt",
                 model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("WARNING: Running on CPU. This will be significantly slower.")

        print(f"Using device: {self.device}")
        print("Loading model and tokenizer...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

        hidden_size = self.model.config.hidden_size
        print(f"Model hidden size: {hidden_size}")

        print("Loading SDL...")
        checkpoint = torch.load(sdl_path, map_location=self.device, weights_only=True)
        dict_size = 4096  # Adjust as needed
        self.sdl = EnhancedSparseAutoencoder(hidden_size, dict_size).to(self.device)

        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                self.sdl.load_state_dict(checkpoint['model'])
            else:
                self.sdl.load_state_dict(checkpoint)
        else:
            self.sdl.load_state_dict(checkpoint)

        self.sdl.eval()
        self.sdl = self.sdl.to(self.device)
        if self.device == "cuda":
            self.sdl.float()
            torch.backends.cudnn.benchmark = True

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print(f"Current VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        # Establish autocast context for mixed precision if available
        self.autocast_context = torch.amp.autocast('cuda',
                                                   dtype=torch.float32) if self.device == "cuda" else nullcontext()

    def analyze_semantic_clusters(self, num_clusters: int = 5):
        """
        Uses k-means clustering on the dictionary atoms to identify semantic groupings.
        Returns an array of cluster labels, one for each atom.
        """
        from sklearn.cluster import KMeans
        # Detach the weights before converting to NumPy
        W = self.sdl.encoder.dictionary.weight.detach().float().cpu().numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(W)
        return clusters

    def compute_feature_importance(self, atom_id: int, text: str):
        """
        Computes the causal impact of a specific dictionary atom on a text.
        Zeros out the atom’s activations and measures the average change (norm difference)
        in the decoded hidden state.
        """
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad(), self.autocast_context:
            # Forward pass to get hidden states from the first layer
            output = self.model(**inputs, output_hidden_states=True)
            hidden_states = output.hidden_states[1]  # shape: (batch, seq_len, hidden_size)
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            # Process through SDL in one batch
            _, base_activations = self.sdl(hidden_states)
            # Clone to create a baseline
            base_activations = base_activations.clone()

            # Zero out the chosen atom
            modified_activations = base_activations.clone()
            modified_activations[:, atom_id] = 0

            # Decode both representations
            base_decoded = self.sdl.decoder(base_activations)
            modified_decoded = self.sdl.decoder(modified_activations)

            # Compute the average difference (impact) across tokens
            impact = torch.norm(base_decoded - modified_decoded, dim=1).mean().item()
        return impact

    def compute_all_feature_importances(self, text: str, atom_indices=None):
        """
        Computes feature importance for multiple atoms on the provided text.
        If atom_indices is None, computes for all atoms in the dictionary.
        Returns a dictionary mapping each atom index to its computed impact.
        """
        # Determine which atoms to test
        total_atoms = self.sdl.encoder.dictionary.weight.shape[0]
        if atom_indices is None:
            atom_indices = list(range(total_atoms))
        impacts = {}
        for atom_id in atom_indices:
            impact = self.compute_feature_importance(atom_id, text)
            impacts[atom_id] = impact
        return impacts

    def visualize_atom_activation(self, atom_id: int, activations: torch.Tensor):
        """
        Utility function to plot the distribution, pattern, and heatmap of activations for a given atom.
        """
        activations_np = activations[:, atom_id].detach().cpu().numpy()
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.hist(activations_np, bins=50, density=True)
        plt.title(f"Atom {atom_id} Activation Distribution")
        plt.xlabel("Activation Value")
        plt.ylabel("Density")

        plt.subplot(1, 3, 2)
        plt.plot(activations_np[:100])
        plt.title(f"Atom {atom_id} Activation Pattern (first 100 tokens)")
        plt.xlabel("Token Position")
        plt.ylabel("Activation Value")

        plt.subplot(1, 3, 3)
        plt.imshow(activations_np[:100].reshape(-1, 1), aspect='auto', cmap='viridis')
        plt.title(f"Atom {atom_id} Activation Heatmap")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


def main():
    check_cuda_availability()

    analyzer = SemanticFeatureAnalyzer(
        sdl_path="layer_1_epoch_49.pt",
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )

    # Example: Cluster dictionary atoms into semantic groups.
    num_clusters = 5
    clusters = analyzer.analyze_semantic_clusters(num_clusters=num_clusters)
    print(f"\nDictionary atoms clustered into {num_clusters} semantic groups:")
    for atom_id, cluster_label in enumerate(clusters):
        print(f"Atom {atom_id}: Cluster {cluster_label}")

    # Example: Compute feature importance for multiple atoms on a sample text.
    sample_text = (
        "Advances in artificial intelligence and machine learning are rapidly transforming society. "
        "New research continues to push the boundaries of what technology can achieve."
    )
    # For example, compute importance for a subset of atoms (e.g., the first 100 atoms)
    atom_indices_to_test = list(range(100))
    impacts = analyzer.compute_all_feature_importances(sample_text, atom_indices=atom_indices_to_test)

    # Optionally, sort and print the atoms by their impact values
    sorted_impacts = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature importances (impact values) for selected atoms:")
    for atom_id, impact in sorted_impacts:
        print(f"Atom {atom_id:4d}: Impact = {impact:.4f}")

    # Optionally, visualize activations for one of the highly impactful atoms.
    chosen_atom = sorted_impacts[0][0]
    print(f"\nVisualizing activations for atom {chosen_atom}")
    inputs = analyzer.tokenizer(sample_text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(analyzer.device) for k, v in inputs.items()}
    with torch.no_grad(), analyzer.autocast_context:
        output = analyzer.model(**inputs, output_hidden_states=True)
        hidden_states = output.hidden_states[1].view(-1, output.hidden_states[1].size(-1))
        _, activations = analyzer.sdl(hidden_states)
    analyzer.visualize_atom_activation(chosen_atom, activations)


if __name__ == "__main__":
    main()
