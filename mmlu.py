import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
import gc
from contextlib import nullcontext

class StiefelGrassmannianDictionary(nn.Module):
    def __init__(self, input_dim: int, dict_size: int, eps: float = 1e-6, tau: float = 0.01):
        super().__init__()
        self.dictionary = nn.Linear(input_dim, dict_size, bias=False)
        nn.init.orthogonal_(self.dictionary.weight)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dictionary(x)

class EnhancedSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.encoder = StiefelGrassmannianDictionary(input_dim, hidden_dim, eps=eps)
        self.decoder = StiefelGrassmannianDictionary(hidden_dim, input_dim, eps=eps)
        self.eps = eps
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = x.float()
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_scaled = x_centered / (torch.norm(x_centered, dim=1, keepdim=True) + self.eps)
        model_dtype = self.encoder.dictionary.weight.dtype
        x_scaled = x_scaled.to(dtype=model_dtype)
        encoded = self.encoder(x_scaled)
        abs_encoded = encoded.float()
        threshold = torch.quantile(abs_encoded.abs(), 0.8, dim=1, keepdim=True)
        mask = (abs_encoded.abs() > threshold).to(dtype=model_dtype)
        activations = encoded * mask
        decoded = self.decoder(activations).float()
        output = decoded * torch.norm(x_centered, dim=1, keepdim=True) + x.mean(dim=1, keepdim=True)
        return output, activations

class MemoryEfficientSDLAnalyzer:
    def __init__(self, sdl_path: str = "checkpoints/layer_1_epoch_49.pt", model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True).to(self.device)
        hidden_size = self.model.config.hidden_size
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
        dict_size = 4096
        self.sdl = EnhancedSparseAutoencoder(hidden_size, dict_size).to(self.device)
        self._load_sdl_state(checkpoint_state)
        self.sdl.eval()
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        self.autocast_context = torch.amp.autocast("cuda", dtype=torch.float32) if self.device == "cuda" else nullcontext()
    def _load_sdl_state(self, checkpoint_state: dict):
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
    def sdl_score_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), self.autocast_context:
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[1]
            orig_shape = hidden.shape
            hidden_flat = hidden.view(-1, hidden.shape[-1])
            modified_hidden_flat, _ = self.sdl(hidden_flat)
            modified_hidden = modified_hidden_flat.view(*orig_shape)
            logits = self.model.lm_head(modified_hidden)
            return logits

def compute_candidate_score(prompt: str, candidate: str, model_forward, model, tokenizer, device):
    full_text = prompt + candidate
    inputs = tokenizer(full_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_length = prompt_ids.shape[1]
    with torch.no_grad():
        logits = model_forward(input_ids, model)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    candidate_ids = input_ids[0, prompt_length:]
    score = 0.0
    for i, token_id in enumerate(candidate_ids):
        score += log_probs[0, prompt_length - 1 + i, token_id].item()
    return score

def base_forward(input_ids, model):
    with torch.no_grad():
        outputs = model(input_ids)
        return outputs.logits

def sdl_forward(input_ids, model_obj):
    return model_obj.sdl_score_text(input_ids)

def evaluate_question(prompt: str, device, model, tokenizer, forward_fn):
    candidates = [" A", " B", " C", " D"]
    scores = {}
    for cand in candidates:
        score = compute_candidate_score(prompt, cand, forward_fn, model, tokenizer, device)
        scores[cand.strip()] = score
    predicted = max(scores, key=scores.get)
    return predicted, scores

def map_correct_answer(ans):
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    try:
        idx = int(ans)
        return mapping.get(idx, str(ans))
    except:
        return str(ans).strip().upper()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    mmlu = load_dataset("cais/mmlu", "all")
    subject_groups = {
        "stem": ["abstract_algebra", "astronomy", "college_chemistry", "physics"],
        "humanities": ["philosophy", "world_religions", "high_school_european_history"],
        "social_science": ["high_school_psychology", "sociology", "public_relations"],
        "other": ["professional_accounting", "business_ethics", "global_facts"]
    }
    analyzer = MemoryEfficientSDLAnalyzer()
    tokenizer = analyzer.tokenizer
    base_model = analyzer.model
    sdl_model_obj = analyzer
    NUM_PER_DOMAIN = 10
    total_questions = 0
    base_correct = 0
    sdl_correct = 0
    for domain, subjects in subject_groups.items():
        domain_questions = []
        for subject in subjects:
            subject_data = mmlu["test"].filter(lambda x: x["subject"] == subject)
            if len(subject_data) > 0:
                domain_questions.extend(subject_data)
        if len(domain_questions) == 0:
            print(f"No questions found for domain {domain}.")
            continue
        sampled = np.random.choice(domain_questions, size=min(NUM_PER_DOMAIN, len(domain_questions)), replace=False)
        for q in sampled:
            prompt = (f"Question: {q['question']}\n"
                      f"Options:\n"
                      f"A) {q['choices'][0]}\n"
                      f"B) {q['choices'][1]}\n"
                      f"C) {q['choices'][2]}\n"
                      f"D) {q['choices'][3]}\n"
                      "Answer:")
            base_pred, base_scores = evaluate_question(prompt, device, base_model, tokenizer, base_forward)
            sdl_pred, sdl_scores = evaluate_question(prompt, device, sdl_model_obj, tokenizer, sdl_forward)
            correct = map_correct_answer(q["answer"])
            total_questions += 1
            if base_pred == correct:
                base_correct += 1
            if sdl_pred == correct:
                sdl_correct += 1
            print(f"Domain: {domain} | Correct: {correct} | Base: {base_pred} ({base_scores}) | SDL: {sdl_pred} ({sdl_scores})")
    base_acc = base_correct / total_questions * 100 if total_questions > 0 else 0.0
    sdl_acc = sdl_correct / total_questions * 100 if total_questions > 0 else 0.0
    print("\nEvaluation Complete!")
    print(f"Total Questions Evaluated: {total_questions}")
    print(f"Base Model Accuracy: {base_acc:.2f}%")
    print(f"SDL-enhanced Model Accuracy: {sdl_acc:.2f}%")

if __name__ == "__main__":
    main()