import torch
import torch.nn as nn
from click.core import F
from transformers import AutoTokenizer, AutoModelForCausalLM, activations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from typing import List, Dict
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.nn.utils.parametrizations import orthogonal
from contextlib import nullcontext
import gc

# ---------------------------
# Model and Utility Classes
# ---------------------------

class StiefelGrassmannianDictionary(nn.Module):
    def __init__(self, input_dim: int, dict_size: int, eps: float = 1e-6, tau: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.tau = tau
        self.eps = eps
        # Wrap a linear layer with an orthogonal parametrization.
        self.dictionary = orthogonal(nn.Linear(input_dim, dict_size, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dictionary(x)


class EnhancedSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.encoder = StiefelGrassmannianDictionary(input_dim, hidden_dim, eps=self.eps)
        self.decoder = StiefelGrassmannianDictionary(hidden_dim, input_dim, eps=self.eps)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Ensure input is in float format and center & scale each sample.
        x = x.float()
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_scaled = x_centered / (torch.norm(x_centered, dim=1, keepdim=True) + self.eps)
        model_dtype = self.encoder.dictionary.weight.dtype
        x_scaled = x_scaled.to(dtype=model_dtype)
        encoded = self.encoder(x_scaled)
        abs_encoded = encoded.float()
        # Apply an 80th-percentile threshold to sparsify activations.
        threshold = torch.quantile(abs_encoded.abs(), 0.8, dim=1, keepdim=True)
        mask = (abs_encoded.abs() > threshold).to(dtype=model_dtype)
        activations = encoded * mask
        decoded = self.decoder(activations).float()
        output = decoded * torch.norm(x_centered, dim=1, keepdim=True) + x.mean(dim=1, keepdim=True)
        return output, activations


# ---------------------------
# Analyzer & Checkpoint Loader
# ---------------------------

class MemoryEfficientSDLAnalyzer:
    def __init__(
        self,
        sdl_path: str = "checkpoints/v2/layer_1_epoch_49.pt",
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the language model with FP16 precision.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)
        hidden_size = self.model.config.hidden_size

        # Load checkpoint (which might be a container with extra keys)
        checkpoint = torch.load(sdl_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                checkpoint_state = checkpoint['model']
            elif 'model_state' in checkpoint:
                checkpoint_state = checkpoint['model_state']
            else:
                checkpoint_state = checkpoint
        else:
            checkpoint_state = checkpoint

        # Create the SDL autoencoder. (Using 4096 atoms/dictionary size.)
        dict_size = 4096
        self.sdl = EnhancedSparseAutoencoder(hidden_size, dict_size).to(self.device)
        # Remap and load checkpoint keys into the current state dict.
        self._load_sdl_state(checkpoint_state)
        self.sdl.eval()
        if self.device == "cuda":
            # Ensure TF32 is enabled for optimal performance on 4090.
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        self.autocast_context = torch.amp.autocast("cuda", dtype=torch.float32) if self.device == "cuda" else nullcontext()

    def _load_sdl_state(self, checkpoint_state: Dict):
        """
        Remap checkpoint keys (which might be saved without parametrizations)
        into the current model's state dict that uses parametrizations.
        In particular, map keys like "encoder.dictionary.weight" into
        "encoder.dictionary.parametrizations.weight.original" and compute a base.
        """
        model_state = self.sdl.state_dict()
        new_state = {}
        for key in model_state.keys():
            if key in checkpoint_state:
                new_state[key] = checkpoint_state[key]
            elif "parametrizations.weight.original" in key:
                base_key = key.replace("parametrizations.weight.original", "weight")
                if base_key in checkpoint_state:
                    new_state[key] = checkpoint_state[base_key]
                else:
                    print(f"Warning: missing key for {key} (tried {base_key})")
            elif "parametrizations.weight.0.base" in key:
                base_key = key.replace("parametrizations.weight.0.base", "weight")
                if base_key in checkpoint_state:
                    pretrained_weight = checkpoint_state[base_key]
                    Q, R = torch.linalg.qr(pretrained_weight.T)
                    new_state[key] = R.T
                else:
                    print(f"Warning: missing key for {key} (tried {base_key})")
            else:
                new_state[key] = model_state[key]
        load_result = self.sdl.load_state_dict(new_state, strict=False)
        print("SDL checkpoint load result:")
        print("  Missing keys:", load_result.missing_keys)
        print("  Unexpected keys:", load_result.unexpected_keys)

    def process_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad(), self.autocast_context:
            output = self.model(**inputs, output_hidden_states=True)
            hidden_states = output.hidden_states[1]
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            batch_size = 32
            activation_batches = []
            for i in range(0, hidden_states.size(0), batch_size):
                chunk = hidden_states[i : i + batch_size]
                _, activations = self.sdl(chunk)
                activation_batches.append(activations.cpu())
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            return torch.cat(activation_batches, dim=0)


# ---------------------------
# Data Analysis & Visualization
# ---------------------------

@dataclass
class AtomAnalysis:
    activation_patterns: torch.Tensor
    temporal_correlations: torch.Tensor
    token_associations: Dict[str, float]
    phase_relationships: torch.Tensor


class EnhancedSDLVisualizer:
    def __init__(self, analyzer: MemoryEfficientSDLAnalyzer):
        self.analyzer = analyzer
        self.color_palette = sns.color_palette("husl", 10)
        # Use the updated seaborn style name.
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            print("Warning: 'seaborn-v0_8-darkgrid' style not found. Falling back to 'seaborn-v0_8'")
            plt.style.use("seaborn-v0_8")

    def analyze_atom_pair(self, atom1: int, atom2: int, text: str) -> AtomAnalysis:
        activations = self.analyzer.process_text(text)
        act1 = activations[:, atom1].cpu()
        act2 = activations[:, atom2].cpu()
        correlation = correlate(act1, act2, mode="full")
        tokens = self.analyzer.tokenizer.encode(text)
        token_texts = self.analyzer.tokenizer.convert_ids_to_tokens(tokens)
        token_associations = {}
        for i, token in enumerate(token_texts):
            if i < len(act1):
                token_associations[token] = ((act1[i] + act2[i]).item()) / 2
        fft1 = torch.fft.fft(act1)
        fft2 = torch.fft.fft(act2)
        phase_diff = torch.angle(fft2) - torch.angle(fft1)
        return AtomAnalysis(
            activation_patterns=torch.stack([act1, act2]),
            temporal_correlations=torch.from_numpy(correlation),
            token_associations=token_associations,
            phase_relationships=phase_diff
        )

    def plot_activation_comparison(self, atom_ids: List[int], text: str):
        activations = self.analyzer.process_text(text)
        tokens = self.analyzer.tokenizer.encode(text)
        token_texts = self.analyzer.tokenizer.convert_ids_to_tokens(tokens)
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Activation Time Series",
                "Activation Distribution",
                "Token-Level Analysis"
            ),
            vertical_spacing=0.1
        )
        # Plot time series traces using distinct colors.
        for idx, atom_id in enumerate(atom_ids):
            r, g, b = [int(255 * c) for c in self.color_palette[idx]]
            line_color = f"rgb({r},{g},{b})"
            fig.add_trace(
                go.Scatter(y=activations[:, atom_id].cpu(), name=f"Atom {atom_id}", line=dict(color=line_color)),
                row=1,
                col=1
            )
        # Plot histograms for activation distributions.
        for idx, atom_id in enumerate(atom_ids):
            fig.add_trace(
                go.Histogram(x=activations[:, atom_id].cpu(), name=f"Atom {atom_id}", nbinsx=50, opacity=0.7),
                row=2,
                col=1
            )
        # Plot a heatmap: rows are atoms, columns are tokens.
        heatmap_data = activations[:, atom_ids].cpu().T
        fig.add_trace(
            go.Heatmap(z=heatmap_data, x=token_texts, y=[f"Atom {aid}" for aid in atom_ids], colorscale="Viridis"),
            row=3,
            col=1
        )
        fig.update_layout(height=1000, title_text="Atom Activation Analysis")
        return fig

    def plot_phase_space(self, atom1: int, atom2: int, text: str):
        activations = self.analyzer.process_text(text)
        act1 = activations[:, atom1].cpu()
        act2 = activations[:, atom2].cpu()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=act1,
                y=act2,
                mode="lines+markers",
                marker=dict(
                    size=8,
                    color=np.arange(len(act1)),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Time Step")
                ),
                name="Phase Trajectory"
            )
        )
        fig.update_layout(
            title=f"Phase Space Analysis: Atom {atom1} vs Atom {atom2}",
            xaxis_title=f"Atom {atom1} Activation",
            yaxis_title=f"Atom {atom2} Activation",
            showlegend=True
        )
        return fig

    def analyze_token_contexts(self, atom_ids: List[int], text: str, window_size: int = 5):
        activations = self.analyzer.process_text(text)
        tokens = self.analyzer.tokenizer.encode(text)
        token_texts = self.analyzer.tokenizer.convert_ids_to_tokens(tokens)
        contexts = []
        # Slide a window over the tokens.
        for i in range(len(tokens) - window_size + 1):
            window_tokens = token_texts[i : i + window_size]
            window_text = self.analyzer.tokenizer.convert_tokens_to_string(window_tokens)
            for atom_id in atom_ids:
                activation = activations[i : i + window_size, atom_id].mean().item()
                if abs(activation) > 0.1:
                    contexts.append({
                        "atom_id": atom_id,
                        "context": window_text,
                        "activation": activation,
                        "position": i
                    })
        return contexts  # returning list of dictionaries instead of a DataFrame

    def analyze_cot_patterns(self, text: str) -> Dict:
        activations = self.analyzer.process_text(text)
        tokens = self.analyzer.tokenizer.encode(text)
        token_texts = self.analyzer.tokenizer.convert_ids_to_tokens(tokens)

        steps = [s.strip() for s in text.split('\n') if s.strip().startswith(('1.', '2.', '3.', '4.'))]
        step_patterns = []
        current_step = []
        step_idx = 0

        for i, token in enumerate(token_texts):
            if i < len(activations):
                if any(step.split()[0] in token for step in steps):
                    if current_step:
                        step_patterns.append({
                            'step': step_idx,
                            'activations': torch.stack(current_step),
                            'tokens': token_texts[i - len(current_step):i]
                        })
                    current_step = []
                    step_idx += 1
                current_step.append(activations[i])

        atom_patterns = self.find_solution_markers(activations)
        return {'steps': step_patterns, 'atoms': atom_patterns}

    def find_solution_markers(self, activations: torch.Tensor) -> List[Dict]:
        threshold = 0.8
        high_activations = (activations > torch.quantile(activations, threshold, dim=0))
        patterns = []

        for i in range(activations.size(1)):
            atom_activations = high_activations[:, i]
            if atom_activations.any():
                points = atom_activations.nonzero().squeeze()
                if len(points.shape) > 0 and points.shape[0] > 1:
                    gaps = points[1:] - points[:-1]
                    if gaps.shape[0] > 0:
                        regularity = float(torch.std(gaps.float()) / (torch.mean(gaps.float()) + 1e-6))
                        patterns.append({
                            'atom': i,
                            'timing': points.tolist(),
                            'avg_gap': float(torch.mean(gaps.float())),
                            'regularity': regularity
                        })
        return patterns

    def plot_cot_analysis(self, atom_patterns: List[Dict], step_patterns: List[Dict]):
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Atom Activation Patterns', 'Step Analysis'))

        for pattern in atom_patterns:
            fig.add_trace(
                go.Scatter(
                    x=pattern['timing'],
                    y=[pattern['atom']] * len(pattern['timing']),
                    mode='markers',
                    name=f'Atom {pattern["atom"]}'
                ),
                row=1, col=1
            )

        for step in step_patterns:
            fig.add_trace(
                go.Heatmap(
                    z=step['activations'].mean(dim=0).unsqueeze(0),
                    name=f'Step {step["step"]}'
                ),
                row=2, col=1
            )

        fig.update_layout(height=800, title_text='Chain of Thought Analysis')
        return fig


# ---------------------------
# Main Benchmarking Script
# ---------------------------

def main():
    from datasets import load_dataset
    import random

    analyzer = MemoryEfficientSDLAnalyzer()
    visualizer = EnhancedSDLVisualizer(analyzer)
    mmlu = load_dataset("cais/mmlu", "all")

    subject_groups = {
        "stem": ["abstract_algebra", "astronomy", "college_chemistry", "physics"],
        "humanities": ["philosophy", "world_religions", "high_school_european_history"],
        "social_science": ["high_school_psychology", "sociology", "public_relations"],
        "other": ["professional_accounting", "business_ethics", "global_facts"]
    }

    results = {}

    for domain, subjects in subject_groups.items():
        chosen_text = None
        chosen_subject = None
        for subject in subjects:
            subject_data = mmlu["test"].filter(lambda x: x["subject"] == subject)
            if len(subject_data) > 0:
                idx = random.randint(0, len(subject_data) - 1)
                question = subject_data[idx]
                chosen_text = (
                    f"Question: {question['question']}\n\n"
                    "Let's solve this step by step:\n"
                    "1. First, let's understand what we're being asked.\n"
                    "2. Now, let's analyze the possible answers:\n"
                    f"A) {question['choices'][0]}\n"
                    f"B) {question['choices'][1]}\n"
                    f"C) {question['choices'][2]}\n"
                    f"D) {question['choices'][3]}\n"
                    "3. Let's think through each option carefully.\n"
                    "4. Finally, we can determine the correct answer."
                )
                chosen_subject = subject
                break

        if chosen_text is not None:
            print(f"\nAnalyzing domain '{domain}' using subject '{chosen_subject}'...")

            # Basic analysis
            fig_activation = visualizer.plot_activation_comparison([0, 4], chosen_text)
            fig_activation.write_html(f"activation_{domain}_{chosen_subject}.html")
            fig_phase = visualizer.plot_phase_space(0, 4, chosen_text)
            fig_phase.write_html(f"phase_{domain}_{chosen_subject}.html")
            token_contexts = visualizer.analyze_token_contexts([0, 4], chosen_text)
            analysis = visualizer.analyze_atom_pair(0, 4, chosen_text)

            # Get base results
            results[domain] = {
                "correlation_peak": analysis.temporal_correlations.max().item(),
                "avg_phase_diff": analysis.phase_relationships.mean().item(),
                "max_activation": analysis.activation_patterns.max().item(),
                "correct_answer": question["answer"],
                "context_activations": token_contexts
            }

            # Add CoT analysis
            cot_results = visualizer.analyze_cot_patterns(chosen_text)
            activations = visualizer.analyzer.process_text(chosen_text)
            solution_markers = visualizer.find_solution_markers(activations)

            results[domain].update({
                'reasoning_steps': len(cot_results['steps']),
                'solution_markers': len(solution_markers),
                'regular_patterns': [p for p in solution_markers if p['regularity'] < 0.2]
            })

            # Plot CoT analysis
            cot_fig = visualizer.plot_cot_analysis(cot_results['atoms'], cot_results['steps'])
            cot_fig.write_html(f"cot_analysis_{domain}_{chosen_subject}.html")
        else:
            print(f"No data found for domain '{domain}'.")

    print("\nAnalysis Complete! Summary of findings:")
    for domain, metrics in results.items():
        print(f"\n{domain}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()