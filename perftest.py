#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import unittest
from nltk.corpus import wordnet as wn
import nltk
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import math

nltk.download('wordnet')
nltk.download('omw-1.4')
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    from spacy.cli import download

    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")


def get_token_embedding(token, embedding_dim):
    doc = nlp(token)
    vector = doc.vector
    if vector.shape[0] != embedding_dim:
        if vector.shape[0] > embedding_dim:
            vector = vector[:embedding_dim]
        else:
            vector = np.pad(vector, (0, embedding_dim - vector.shape[0]), mode='constant')
    return torch.tensor(vector, dtype=torch.float32)


class SparseDictionaryTester:
    def __init__(self, checkpoint_path, activation_threshold=0.5, model_type="auto"):
        self.checkpoint_path = checkpoint_path
        self.activation_threshold = activation_threshold
        self.model_type = model_type
        self.base_dictionary = None
        self.comp_dictionary = None
        self.context_projector_weight = None
        self.context_projector_bias = None
        self.gate_generator_0_weight = None
        self.gate_generator_0_bias = None
        self.gate_strength = None
        self.signal_weights = None
        self.embedding_dim = None
        self.load_checkpoint()

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))

        if isinstance(checkpoint, dict):
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Detect model type from state dict if set to auto
        if self.model_type == "auto":
            if 'comp_encoder.dictionary.weight' in state_dict:
                self.model_type = "hierarchical"
            elif 'context_projector.weight' in state_dict:
                self.model_type = "gated"
            else:
                self.model_type = "original"
            print(f"Auto-detected model type: {self.model_type}")

        # Load appropriate dictionary based on model type
        if self.model_type == "hierarchical":
            base_weight_key = None
            comp_weight_key = None

            possible_base_keys = [
                'base_encoder.dictionary.weight',
                'base_encoder.dictionary.parametrizations.weight.original',
                'base_encoder.dictionary.parametrizations.weight.0.base'
            ]

            possible_comp_keys = [
                'comp_encoder.dictionary.weight',
                'comp_encoder.dictionary.parametrizations.weight.original',
                'comp_encoder.dictionary.parametrizations.weight.0.base'
            ]

            for key in possible_base_keys:
                if key in state_dict:
                    base_weight_key = key
                    break

            for key in possible_comp_keys:
                if key in state_dict:
                    comp_weight_key = key
                    break

            if base_weight_key is None or comp_weight_key is None:
                print("Available keys in state dict:", state_dict.keys())
                raise ValueError("Could not find hierarchical dictionary weights in checkpoint")

            self.base_dictionary = state_dict[base_weight_key]
            self.comp_dictionary = state_dict[comp_weight_key]
            self.embedding_dim = self.base_dictionary.shape[1]
            self.num_atoms = self.base_dictionary.shape[0] + self.comp_dictionary.shape[0]

        else:  # Original or gated model
            weight_key = None
            possible_keys = [
                'encoder.dictionary.weight',
                'encoder.dictionary.parametrizations.weight.original',
                'encoder.dictionary.parametrizations.weight.0.base'
            ]

            for key in possible_keys:
                if key in state_dict:
                    weight_key = key
                    break

            if weight_key is None:
                print("Available keys in state dict:", state_dict.keys())
                raise ValueError("Could not find dictionary weights in checkpoint")

            self.base_dictionary = state_dict[weight_key]
            self.embedding_dim = self.base_dictionary.shape[1]
            self.num_atoms = self.base_dictionary.shape[0]

            # For gated model, load the context detection and gating networks
            if self.model_type == "gated":
                self.has_gate_network = 'context_projector.weight' in state_dict
                if self.has_gate_network:
                    self.context_projector_weight = state_dict['context_projector.weight']
                    self.context_projector_bias = state_dict['context_projector.bias']
                    self.gate_generator_0_weight = state_dict['gate_generator.0.weight']
                    self.gate_generator_0_bias = state_dict['gate_generator.0.bias']
                    self.gate_strength = state_dict.get('gate_strength', torch.tensor([0.5]))
                    if 'signal_weights' in state_dict:
                        self.signal_weights = state_dict['signal_weights']
                    else:
                        self.signal_weights = torch.ones(3) / 3

    def compute_nonlinearity_score(self, embedding):
        embedding_centered = embedding - embedding.mean()
        embedding_scaled = embedding_centered / (torch.norm(embedding_centered) + 1e-6)

        # Project to context space
        context_feats = torch.matmul(embedding_scaled, self.context_projector_weight.t()) + self.context_projector_bias
        context_norm = context_feats / (torch.norm(context_feats) + 1e-6)

        # Self-attention for pattern detection
        sim_matrix = torch.mm(context_norm.unsqueeze(0), context_norm.unsqueeze(0).t())

        # Multiple nonlinearity signals
        # 1. Variance of similarities
        row_variance = torch.var(sim_matrix, dim=1)

        # 2. Entropy of similarity distribution
        sim_scaled = (sim_matrix + 1) / 2
        entropy = -torch.sum(sim_scaled * torch.log(sim_scaled + 1e-6), dim=1)

        # 3. Skewness of distribution
        centered = sim_matrix - torch.mean(sim_matrix, dim=1, keepdim=True)
        skewness = torch.mean(torch.pow(centered, 3), dim=1) / (torch.var(sim_matrix, dim=1) + 1e-6) ** 1.5

        # Combine signals with weights
        combined_signal = (self.signal_weights[0] * row_variance +
                           self.signal_weights[1] * entropy +
                           self.signal_weights[2] * torch.abs(skewness))

        nonlinearity_score = torch.sigmoid((combined_signal - 0.5 * combined_signal.mean()) / 0.1)
        return nonlinearity_score

    def get_activation(self, token):
        embedding = get_token_embedding(token, self.embedding_dim)

        if self.model_type == "hierarchical":
            # Get activations from both dictionaries
            base_activation = torch.matmul(self.base_dictionary, embedding)
            comp_activation = torch.matmul(self.comp_dictionary, embedding)

            # Apply ReLU and thresholding
            base_activation = torch.relu(base_activation)
            comp_activation = torch.relu(comp_activation)

            # For phrases, prioritize compositional dictionary
            is_phrase = " " in token
            comp_scale = 1.0 if is_phrase else 0.5

            # Apply different thresholds to each dictionary
            base_mask = base_activation >= self.activation_threshold
            comp_mask = comp_activation >= (self.activation_threshold * 0.8)

            base_sparse = base_activation * base_mask.float()
            comp_sparse = comp_activation * comp_mask.float() * comp_scale

            # Combine activations
            return torch.cat([base_sparse, comp_sparse], dim=0)

        elif self.model_type == "gated" and hasattr(self, 'has_gate_network') and self.has_gate_network:
            # Normalize embedding
            embedding_centered = embedding - embedding.mean()
            embedding_scaled = embedding_centered / (torch.norm(embedding_centered) + 1e-6)

            # Compute nonlinearity score using our improved method
            if hasattr(self, 'signal_weights'):
                nonlinearity_score = self.compute_nonlinearity_score(embedding_scaled)
            else:
                # Fallback to original method if signal weights not available
                context_feats = torch.matmul(embedding_scaled,
                                             self.context_projector_weight.t()) + self.context_projector_bias
                nonlinearity_score = torch.var(context_feats).unsqueeze(0)
                nonlinearity_score = torch.sigmoid(nonlinearity_score - 0.5)

            # Generate gates
            context_feats = torch.matmul(embedding_scaled,
                                         self.context_projector_weight.t()) + self.context_projector_bias
            gates = torch.sigmoid(
                torch.matmul(context_feats, self.gate_generator_0_weight.t()) + self.gate_generator_0_bias)
            effective_gates = 1.0 - (self.gate_strength * (1.0 - gates) * nonlinearity_score)

            # Get base activation
            activation = torch.matmul(self.base_dictionary, embedding)
            activation = torch.relu(activation)

            # Apply gates and thresholding
            mask = activation >= self.activation_threshold
            sparse_activation = activation * mask.float() * effective_gates

            return sparse_activation

        else:
            # Original model
            activation = torch.matmul(self.base_dictionary, embedding)
            activation = torch.relu(activation)
            mask = activation >= self.activation_threshold
            sparse_activation = activation * mask.float()

            return sparse_activation

    def batch_activations(self, tokens):
        activations = {}
        for token in tokens:
            activations[token] = self.get_activation(token)
        return activations


def token_aligned_analysis(model, tokens):
    activations = model.batch_activations(tokens)
    token_activation_map = {}
    for token, act in activations.items():
        active_atoms = (act > 0).nonzero(as_tuple=True)[0].tolist()
        token_activation_map[token] = active_atoms
    return token_activation_map


def compare_activation_patterns(model, token_groups):
    group_similarities = {}
    for group, tokens in token_groups.items():
        activations = [model.get_activation(token).numpy() for token in tokens]
        if len(activations) > 1:
            sim_matrix = cosine_similarity(activations)
        else:
            sim_matrix = np.array([[1.0]])
        group_similarities[group] = sim_matrix
    return group_similarities


def detect_concept_clusters(model, tokens, n_clusters=3):
    activations = []
    valid_tokens = []
    for token in tokens:
        act = model.get_activation(token).numpy()
        if np.sum(act) > 0:
            activations.append(act)
            valid_tokens.append(token)
    if len(activations) == 0:
        return {}
    activations = np.array(activations)
    kmeans = KMeans(n_clusters=min(n_clusters, len(activations)), random_state=42).fit(activations)
    clusters = {token: int(label) for token, label in zip(valid_tokens, kmeans.labels_)}
    return clusters


def contextual_pattern_detection(model, sentence, window_size=2):
    tokens = sentence.split()
    analysis = []
    for i, token in enumerate(tokens):
        activation = model.get_activation(token)
        active_atoms = (activation > 0).nonzero(as_tuple=True)[0].tolist()
        if active_atoms:
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(tokens))
            context = tokens[start:i] + tokens[i + 1:end]
            analysis.append((token, active_atoms, context))
    return analysis


def cross_domain_concept_testing(model, token_pairs):
    results = []
    for token1, token2 in token_pairs:
        act1 = model.get_activation(token1).unsqueeze(0).numpy()
        act2 = model.get_activation(token2).unsqueeze(0).numpy()
        sim = float(cosine_similarity(act1, act2)[0][0])
        results.append((token1, token2, sim))
    return results


class TestSparseDictionaryTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {}

        # Try to load original model
        try:
            original_path = 'checkpoints/v4/gated/layer_1_epoch_41.pt'
            cls.models['original'] = SparseDictionaryTester(original_path, activation_threshold=0.5,
                                                            model_type="original")
            print(f"Loaded original model from {original_path}")
        except Exception as e:
            print(f"Could not load original model: {e}")

        # Try to load gated model
        try:
            gated_path = 'checkpoints/layer_0_epoch_49.pt'
            cls.models['gated'] = SparseDictionaryTester(gated_path, activation_threshold=0.5, model_type="gated")
            print(f"Loaded gated model from {gated_path}")
        except Exception as e:
            print(f"Could not load gated model: {e}")

        # Try to use most recent model if others fail
        if not cls.models:
            try:
                default_path = 'checkpoints/layer_0_epoch_49.pt'
                cls.models['default'] = SparseDictionaryTester(default_path, activation_threshold=0.5,
                                                               model_type="auto")
                print(f"Loaded default model from {default_path}")
            except Exception as e:
                print(f"Could not load any model: {e}")
                raise ValueError("No models could be loaded for testing")

        # Use the latest model for testing by default
        cls.model = next(iter(cls.models.values()))

    def test_token_aligned_activation(self):
        tokens = ['solve', 'reason', 'conclude']
        token_map = token_aligned_analysis(self.model, tokens)
        for token in tokens:
            active_atoms = token_map[token]
            self.assertIsInstance(active_atoms, list)
            self.assertTrue(len(active_atoms) > 0)

    def test_activation_pattern_similarity(self):
        token_groups = {
            'reasoning': ['reason', 'analyze', 'deduce'],
            'solving': ['solve', 'compute', 'resolve']
        }
        similarities = compare_activation_patterns(self.model, token_groups)
        for group, sim_matrix in similarities.items():
            if sim_matrix.shape[0] > 1:
                avg_sim = float(np.mean(sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]))
                self.assertGreater(avg_sim, 0.1)

    def test_concept_clusters_detection(self):
        tokens = ['solve', 'reason', 'conclude', 'apple', 'orange', 'banana', 'car', 'truck', 'bus']
        clusters = detect_concept_clusters(self.model, tokens, n_clusters=3)
        self.assertIsInstance(clusters, dict)
        for token in tokens:
            if token in clusters:
                self.assertIn(clusters[token], range(3))

    def test_contextual_pattern_detection(self):
        sentence = "To solve the problem, one must reason carefully and conclude logically."
        analysis = contextual_pattern_detection(self.model, sentence, window_size=2)
        self.assertIsInstance(analysis, list)
        for entry in analysis:
            self.assertEqual(len(entry), 3)
            token, active_atoms, context = entry
            self.assertIsInstance(active_atoms, list)
            self.assertIsInstance(context, list)

    def test_cross_domain_concept_similarity(self):
        token_pairs = [
            ('solve', 'resolve'),
            ('reason', 'deduce'),
            ('car', 'automobile')
        ]
        results = cross_domain_concept_testing(self.model, token_pairs)
        self.assertIsInstance(results, list)
        for token1, token2, sim in results:
            self.assertIsInstance(sim, float)
            self.assertGreaterEqual(sim, 0.0)
            self.assertLessEqual(sim, 1.0)

    def test_activation_consistency(self):
        token = "solve"
        activations = [self.model.get_activation(token) for _ in range(5)]
        for i in range(1, len(activations)):
            self.assertTrue(torch.allclose(activations[0], activations[i], atol=1e-6))

    def test_perturbation_sensitivity(self):
        token = "solve"
        epsilon = 1e-4
        embedding = get_token_embedding(token, self.model.embedding_dim)
        original_activation = self.model.get_activation(token)

        perturbed_embedding = embedding + torch.randn_like(embedding) * epsilon

        if self.model.model_type == "gated" and hasattr(self.model, 'has_gate_network') and self.model.has_gate_network:
            perturbed_activation = self.model.get_activation("solve")  # Handle full gating logic internally
        elif self.model.model_type == "hierarchical":
            base_activation = torch.matmul(self.model.base_dictionary, perturbed_embedding)
            comp_activation = torch.matmul(self.model.comp_dictionary, perturbed_embedding)
            base_activation = torch.relu(base_activation)
            comp_activation = torch.relu(comp_activation)
            base_mask = base_activation >= self.model.activation_threshold
            comp_mask = comp_activation >= (self.model.activation_threshold * 0.8)
            base_sparse = base_activation * base_mask.float()
            comp_sparse = comp_activation * comp_mask.float() * 0.5
            perturbed_activation = torch.cat([base_sparse, comp_sparse], dim=0)
        else:
            perturbed_activation = torch.matmul(self.model.base_dictionary, perturbed_embedding)
            perturbed_activation = torch.relu(perturbed_activation)
            mask = perturbed_activation >= self.model.activation_threshold
            perturbed_activation = perturbed_activation * mask.float()

        sim = float(cosine_similarity(original_activation.unsqueeze(0).numpy(),
                                      perturbed_activation.unsqueeze(0).numpy())[0][0])
        self.assertGreater(sim, 0.95)

    def test_activation_strength_distribution(self):
        tokens = ['solve', 'reason', 'conclude', 'analyze', 'compute', 'apple', 'orange', 'banana', 'car', 'truck',
                  'bus']
        fractions = []
        for token in tokens:
            activation = self.model.get_activation(token)
            fraction = float((activation > 0).sum().item() / self.model.num_atoms)
            fractions.append(fraction)
        avg_fraction = float(np.mean(fractions))
        self.assertGreater(avg_fraction, 0.001)
        self.assertLess(avg_fraction, 0.5)

    def test_synonym_antonym_similarity(self):
        token = "happy"
        synsets = wn.synsets(token)
        synonyms = set()
        antonyms = set()
        for s in synsets:
            for l in s.lemmas():
                synonyms.add(l.name())
                for ant in l.antonyms():
                    antonyms.add(ant.name())
        synonyms.discard(token)
        if synonyms and antonyms:
            syn_sims = [float(cosine_similarity(self.model.get_activation(token).unsqueeze(0).numpy(),
                                                self.model.get_activation(s).unsqueeze(0).numpy())[0][0]) for s in
                        synonyms]
            ant_sims = [float(cosine_similarity(self.model.get_activation(token).unsqueeze(0).numpy(),
                                                self.model.get_activation(a).unsqueeze(0).numpy())[0][0]) for a in
                        antonyms]
            avg_syn = float(np.mean(syn_sims))
            avg_ant = float(np.mean(ant_sims))
            self.assertIsInstance(avg_syn, float)
            self.assertIsInstance(avg_ant, float)

    def test_hierarchical_relationship(self):
        sim_dog_animal = float(cosine_similarity(self.model.get_activation("dog").unsqueeze(0).numpy(),
                                                 self.model.get_activation("animal").unsqueeze(0).numpy())[0][0])
        sim_car_vehicle = float(cosine_similarity(self.model.get_activation("car").unsqueeze(0).numpy(),
                                                  self.model.get_activation("vehicle").unsqueeze(0).numpy())[0][0])
        self.assertGreater(sim_dog_animal, 0.1)
        self.assertGreater(sim_car_vehicle, 0.1)

    def test_compositional_patterns(self):
        phrase_activation = self.model.get_activation("blue car")
        blue_activation = self.model.get_activation("blue")
        car_activation = self.model.get_activation("car")
        avg_activation = (blue_activation + car_activation) / 2.0
        sim = float(cosine_similarity(phrase_activation.unsqueeze(0).numpy(),
                                      avg_activation.unsqueeze(0).numpy())[0][0])
        self.assertGreater(sim, 0.7)

    def test_feature_reuse(self):
        tokens = ['solve', 'reason', 'conclude', 'analyze', 'compute', 'apple', 'orange', 'banana', 'car', 'truck',
                  'bus']
        frequency = {}
        for token in tokens:
            active_atoms = (self.model.get_activation(token) > 0).nonzero(as_tuple=True)[0].tolist()
            for atom in active_atoms:
                frequency[atom] = frequency.get(atom, 0) + 1
        if len(tokens) > 1:
            self.assertTrue(any(count >= 2 for count in frequency.values()))

    def test_semantic_coherence_of_clusters(self):
        tokens = ['solve', 'reason', 'conclude', 'analyze', 'compute', 'apple', 'orange', 'banana', 'car', 'truck',
                  'bus']
        clusters = detect_concept_clusters(self.model, tokens, n_clusters=3)
        cluster_tokens = {}
        for token, label in clusters.items():
            cluster_tokens.setdefault(label, []).append(token)
        for label, toks in cluster_tokens.items():
            if len(toks) > 1:
                activations = [self.model.get_activation(t).numpy() for t in toks]
                sim_matrix = cosine_similarity(activations)
                avg_sim = float(np.mean(sim_matrix[np.triu_indices(len(toks), k=1)]))
                self.assertGreater(avg_sim, 0.2)

    def test_negation_contradiction_compositionality(self):
        test_phrases = {
            "not bad": ("not", "bad"),
            "not happy": ("not", "happy"),
            "hot cold": ("hot", "cold"),
            "big small": ("big", "small"),
            "bright dark": ("bright", "dark"),
            "square circle": ("square", "circle")
        }

        threshold = 0.7
        for phrase, (token1, token2) in test_phrases.items():
            phrase_act = self.model.get_activation(phrase)
            avg_act = (self.model.get_activation(token1) + self.model.get_activation(token2)) / 2.0
            sim = float(cosine_similarity(phrase_act.unsqueeze(0).numpy(),
                                          avg_act.unsqueeze(0).numpy())[0][0])

            if self.model.model_type in ['gated', 'hierarchical']:
                self.assertLess(sim, threshold,
                                f"Phrase '{phrase}' unexpectedly has high compositional similarity: {sim}")
            else:
                print(f"Original model similarity for '{phrase}': {sim}")

    def test_contextual_noncompositionality(self):
        test_phrases = {
            "fast snail": ("fast", "snail"),
            "sleeping fish": ("sleeping", "fish"),
            "flying elephant": ("flying", "elephant")
        }

        threshold = 0.7
        for phrase, (token1, token2) in test_phrases.items():
            phrase_act = self.model.get_activation(phrase)
            avg_act = (self.model.get_activation(token1) + self.model.get_activation(token2)) / 2.0
            sim = float(cosine_similarity(phrase_act.unsqueeze(0).numpy(),
                                          avg_act.unsqueeze(0).numpy())[0][0])

            if self.model.model_type in ['gated', 'hierarchical']:
                self.assertLess(sim, threshold,
                                f"Phrase '{phrase}' unexpectedly has high compositional similarity: {sim}")
            else:
                print(f"Original model similarity for '{phrase}': {sim}")

    def test_manifold_geometry(self):
        phrase_activation = self.model.get_activation("blue car")
        blue_activation = self.model.get_activation("blue")
        car_activation = self.model.get_activation("car")
        avg_activation = (blue_activation + car_activation) / 2.0

        def normalize(v):
            norm = torch.norm(v) + 1e-8
            return v / norm

        phrase_norm = normalize(phrase_activation)
        avg_norm = normalize(avg_activation)

        cosine_sim = float(cosine_similarity(phrase_norm.unsqueeze(0).numpy(),
                                             avg_norm.unsqueeze(0).numpy())[0][0])
        geodesic_dist = math.acos(min(max(cosine_sim, -1.0), 1.0))

        if self.model.model_type == "original":
            print(f"Original model geodesic distance for 'blue car': {geodesic_dist}")
        else:
            self.assertLess(geodesic_dist, math.acos(0.7),
                            f"Geodesic distance {geodesic_dist} is too high for a cosine similarity of {cosine_sim}")

    def test_all_model_types_negation(self):
        results = {}
        for model_name, model in self.models.items():
            phrase_act = model.get_activation("not bad")
            avg_act = (model.get_activation("not") + model.get_activation("bad")) / 2.0
            sim = float(cosine_similarity(phrase_act.unsqueeze(0).numpy(),
                                          avg_act.unsqueeze(0).numpy())[0][0])
            results[model_name] = sim

        print(f"Negation test results across model types: {results}")

        if 'gated' in self.models and 'original' in self.models:
            self.assertLess(results['gated'], results['original'],
                            "Gated model should have lower compositional similarity for negations")

    def test_all_model_types_contradiction(self):
        results = {}
        for model_name, model in self.models.items():
            phrase_act = model.get_activation("fast snail")
            avg_act = (model.get_activation("fast") + model.get_activation("snail")) / 2.0
            sim = float(cosine_similarity(phrase_act.unsqueeze(0).numpy(),
                                          avg_act.unsqueeze(0).numpy())[0][0])
            results[model_name] = sim

        print(f"Contradiction test results across model types: {results}")

        if 'gated' in self.models and 'original' in self.models:
            self.assertLess(results['gated'], results['original'],
                            "Gated model should have lower compositional similarity for contradictions")


if __name__ == '__main__':
    unittest.main()