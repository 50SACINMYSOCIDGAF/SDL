import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm
import gc
from torch.nn.utils.parametrizations import orthogonal
from contextlib import nullcontext


def check_cuda_availability():
    print("CUDA Environment Check:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("\nCUDA is not available. Checking potential issues:")
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
        x = x.float()
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_scaled = x_centered / (torch.norm(x_centered, dim=1, keepdim=True) + self.eps)

        model_dtype = self.encoder.dictionary.weight.dtype
        x_scaled = x_scaled.to(dtype=model_dtype)

        encoded = self.encoder(x_scaled)

        # Apply a sparsity threshold: keep only the top 20% activations (by absolute value)
        abs_encoded = encoded.float()
        threshold = torch.quantile(abs_encoded.abs(), 0.8, dim=1, keepdim=True)
        mask = (abs_encoded.abs() > threshold).to(dtype=model_dtype)
        activations = encoded * mask

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


class MemoryEfficientSDLAnalyzer:
    def __init__(self,
                 sdl_path: str = "layer_1_epoch_49.pt",
                 model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("WARNING: Running on CPU. This will be significantly slower.")

        print(f"Using device: {self.device}")
        print("Loading model and tokenizer...")

        # Load model first to get correct dimensions
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)

        print(f"Model hidden size: {self.model.config.hidden_size}")
        hidden_size = self.model.config.hidden_size  # e.g. 1536

        print("Loading SDL...")
        checkpoint = torch.load(sdl_path, map_location=self.device, weights_only=True)

        # Initialize the SDL model with correct dimensions from model
        dict_size = 4096  # Your dictionary size for layer 1
        self.sdl = EnhancedSparseAutoencoder(hidden_size, dict_size).to(self.device)

        # Load the state dictionary
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                self.sdl.load_state_dict(checkpoint['model'])
            else:
                self.sdl.load_state_dict(checkpoint)

        self.sdl.eval()

        # Remove explicit half() conversion for SDL
        self.sdl = self.sdl.to(self.device)
        if self.device == "cuda":
            # Maintain float32 precision for numerical stability
            self.sdl.float()
            # Enable cudnn benchmarking for optimized performance
            torch.backends.cudnn.benchmark = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Current VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        print("Loading model and checking dimensions...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)

        print(f"Model hidden size: {self.model.config.hidden_size}")

        checkpoint = torch.load(sdl_path, map_location=self.device, weights_only=True)
        if 'model' in checkpoint:
            input_dim = checkpoint['model']['encoder.dictionary.parametrizations.weight.original'].shape[1]
        else:
            input_dim = checkpoint['encoder.dictionary.parametrizations.weight.original'].shape[1]
        print(f"SDL input dimension: {input_dim}")

        # Add precision context management
        self.autocast_context = torch.amp.autocast('cuda',
                                                   dtype=torch.float32) if self.device == "cuda" else nullcontext()

    def process_text(self, text: str, chunk_size: int = 512):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=chunk_size)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad(), self.autocast_context:
            output = self.model(**inputs, output_hidden_states=True)
            hidden_states = output.hidden_states[1]  # Get the first layer's hidden states

            # Reshape to (num_tokens, hidden_size)
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))

            # Process through SDL in chunks
            chunk_size = 32
            outputs = []

            for i in range(0, hidden_states.size(0), chunk_size):
                chunk = hidden_states[i:i + chunk_size]
                _, activations = self.sdl(chunk)
                outputs.append(activations.cpu())
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        return torch.cat(outputs, dim=0)

    def analyze_dictionary(self, num_atoms: int = 10, test_texts: list = None):
        if test_texts is None:
            test_texts = [
                "The artificial intelligence revolution has transformed modern technology.",
                "Scientific discoveries continue to expand our understanding of the universe.",
                "The integration of machine learning in healthcare has improved patient outcomes."
            ]

        results = []
        all_activations = []

        print("\nProcessing texts and analyzing dictionary atoms...")
        for text in tqdm(test_texts):
            activations = self.process_text(text)
            all_activations.append(activations)

        all_activations = torch.cat(all_activations, dim=0)

        # Move dictionary weights to CPU for analysis
        W = self.sdl.encoder.dictionary.weight.float().cpu()

        for atom_id in range(min(num_atoms, W.size(0))):
            atom_activations = all_activations[:, atom_id]
            threshold = torch.quantile(atom_activations.abs(), 0.8)
            activation_freq = (atom_activations.abs() > threshold).float().mean()

            atom_vector = W[atom_id]
            other_atoms = torch.cat([W[:atom_id], W[atom_id + 1:]])
            orthogonality = torch.max(torch.abs(atom_vector @ other_atoms.t())).item()

            correlations = torch.nn.functional.cosine_similarity(
                atom_vector.unsqueeze(0), other_atoms, dim=1)
            top_k = 5
            values, indices = torch.topk(correlations.abs(), min(top_k, len(correlations)))
            top_correlations = {int(idx): float(val) for val, idx in zip(values, indices)}

            coherence = values[0].item()

            analysis = DictionaryAnalysis(
                atom_id=atom_id,
                activation_frequency=activation_freq.item(),
                orthogonality_measure=orthogonality,
                coherence_score=coherence,
                top_correlations=top_correlations
            )
            results.append(analysis)

            self.visualize_atom(atom_id, atom_activations)

        return results

    def visualize_atom(self, atom_id: int, activations: torch.Tensor):
        plt.figure(figsize=(15, 5))
        activations_np = activations.numpy()

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
        plt.close()

    def compute_feature_importance(self, atom_id: int, text: str):
        """Analyse the causal impact of a specific dictionary atom."""
        with torch.no_grad(), self.autocast_context:
            base_activations = self.process_text(text)
            modified_activations = base_activations.clone()
            modified_activations[:, atom_id] = 0
            impact = torch.norm(base_activations - modified_activations, dim=1)
            return impact.mean().item()

    def analyze_semantic_clusters(self, num_clusters: int = 5):
        """Identify semantic groupings within dictionary atoms using k-means clustering."""
        from sklearn.cluster import KMeans
        W = self.sdl.encoder.dictionary.weight.float().cpu().numpy()
        kmeans = KMeans(n_clusters=num_clusters)
        clusters = kmeans.fit_predict(W)
        return clusters

    def steer_and_generate(self, prompt: str, steering_instructions: dict, max_new_tokens: int = 1):
        """
        Generate tokens with a steer on the hidden representation.

        steering_instructions: dict mapping atom index to a multiplicative steering factor.
          For example, {123: 1.5, 456: 0.5} will amplify the 123rd atom and dampen the 456th.

        In this simple implementation, we only modify the last token’s hidden state from the first layer,
        decode it via the SDL, and then pass it through the LM head to generate the next token.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_tokens = []
        current_prompt = prompt

        with torch.no_grad(), self.autocast_context:
            for _ in range(max_new_tokens):
                # Forward pass to get hidden states
                output = self.model(**inputs, output_hidden_states=True)
                # Use the first layer's hidden states
                hidden_states = output.hidden_states[1]  # shape: (batch, seq_len, hidden_size)
                # Select the last token's hidden state
                last_hidden = hidden_states[:, -1, :]  # shape: (batch, hidden_size)

                # Pass through the SDL encoder to get dictionary activations
                encoded = self.sdl.encoder(last_hidden)
                modified_encoded = encoded.clone()

                # Apply steering: multiply selected atom activations by provided factors
                for atom_idx, factor in steering_instructions.items():
                    modified_encoded[:, atom_idx] = modified_encoded[:, atom_idx] * factor

                # Decode the modified activations to get a modified hidden representation
                modified_hidden = self.sdl.decoder(modified_encoded)

                # Get logits from the language modeling head using the modified hidden state
                logits = self.model.lm_head(modified_hidden)
                # Sample the next token (using argmax for simplicity; you can replace this with more sophisticated sampling)
                next_token_id = torch.argmax(logits, dim=-1)
                generated_tokens.append(next_token_id.item())

                # Append the generated token to the prompt for subsequent generation
                next_token = self.tokenizer.decode(next_token_id)
                current_prompt += next_token

                # Prepare new inputs by tokenizing the updated prompt
                inputs = self.tokenizer(current_prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text, generated_tokens


def main():
    check_cuda_availability()

    try:
        analyzer = MemoryEfficientSDLAnalyzer()

        test_texts = [
            """Artificial intelligence and machine learning have revolutionized modern technology,
            enabling unprecedented advances in automation and data analysis.""",
            """The development of deep learning models has created new possibilities for
            natural language processing and computer vision.""",
            """Neural networks and deep learning architectures continue to evolve,
            pushing the boundaries of what's possible in AI."""
        ]

        results = analyzer.analyze_dictionary(num_atoms=10, test_texts=test_texts)

        print("\nDictionary Analysis Results:")
        for analysis in results:
            print(f"\nAtom {analysis.atom_id}:")
            print(f"Activation Frequency: {analysis.activation_frequency:.4f}")
            print(f"Orthogonality Measure: {analysis.orthogonality_measure:.4f}")
            print(f"Coherence Score: {analysis.coherence_score:.4f}")
            print("Top Correlations:")
            for other_id, corr in analysis.top_correlations.items():
                print(f"  Atom {other_id}: {corr:.4f}")

        # Demonstrate steering
        prompt = "In a surprising turn of events, the breakthrough in technology"
        # For example, let’s try to boost the activation of atom 123 and dampen atom 456
        steering_instructions = {123: 1.5, 456: 0.5}
        steered_text, token_ids = analyzer.steer_and_generate(prompt, steering_instructions, max_new_tokens=5)
        print("\nSteered Generation Result:")
        print(f"Prompt: {prompt}")
        print(f"Generated continuation: {steered_text}")
        print(f"Token IDs: {token_ids}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
