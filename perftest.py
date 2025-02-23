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
    def __init__(self, checkpoint_path, activation_threshold=0.5):
        self.checkpoint_path = checkpoint_path
        self.activation_threshold = activation_threshold
        self.dictionary = None
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
            print("Available keys in state dict:", state_dict.keys())  # Debug info
            raise ValueError("Could not find dictionary weights in checkpoint")

        self.dictionary = state_dict[weight_key]
        self.embedding_dim = self.dictionary.shape[1]
        self.num_atoms = self.dictionary.shape[0]

    def get_activation(self, token):
        embedding = get_token_embedding(token, self.embedding_dim)
        activation = torch.matmul(self.dictionary, embedding)
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
        sim = cosine_similarity(act1, act2)[0][0]
        results.append((token1, token2, sim))
    return results


class TestSparseDictionaryTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_path = 'checkpoints/v2/layer_1_epoch_49.pt'
        print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            print("Checkpoint structure:", checkpoint.keys() if isinstance(checkpoint, dict) else "not a dict")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
        cls.model = SparseDictionaryTester(checkpoint_path, activation_threshold=0.5)

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
                avg_sim = np.mean(sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)])
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
        perturbed_activation = torch.matmul(self.model.dictionary, perturbed_embedding)
        perturbed_activation = torch.relu(perturbed_activation)
        mask = perturbed_activation >= self.model.activation_threshold
        perturbed_activation = perturbed_activation * mask.float()
        sim = cosine_similarity(original_activation.unsqueeze(0).numpy(), perturbed_activation.unsqueeze(0).numpy())[0][
            0]
        self.assertGreater(sim, 0.95)

    def test_activation_strength_distribution(self):
        tokens = ['solve', 'reason', 'conclude', 'analyze', 'compute', 'apple', 'orange', 'banana', 'car', 'truck',
                  'bus']
        fractions = []
        for token in tokens:
            activation = self.model.get_activation(token)
            fraction = (activation > 0).sum().item() / self.model.num_atoms
            fractions.append(fraction)
        avg_fraction = np.mean(fractions)
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
            syn_sims = [cosine_similarity(self.model.get_activation(token).unsqueeze(0).numpy(),
                                          self.model.get_activation(s).unsqueeze(0).numpy())[0][0] for s in synonyms]
            ant_sims = [cosine_similarity(self.model.get_activation(token).unsqueeze(0).numpy(),
                                          self.model.get_activation(a).unsqueeze(0).numpy())[0][0] for a in antonyms]
            avg_syn = np.mean(syn_sims)
            avg_ant = np.mean(ant_sims)
            self.assertGreater(avg_syn, avg_ant)

    def test_hierarchical_relationship(self):
        sim_dog_animal = cosine_similarity(self.model.get_activation("dog").unsqueeze(0).numpy(),
                                           self.model.get_activation("animal").unsqueeze(0).numpy())[0][0]
        sim_car_vehicle = cosine_similarity(self.model.get_activation("car").unsqueeze(0).numpy(),
                                            self.model.get_activation("vehicle").unsqueeze(0).numpy())[0][0]
        self.assertGreater(sim_dog_animal, 0.3)
        self.assertGreater(sim_car_vehicle, 0.3)

    def test_compositional_patterns(self):
        phrase_activation = self.model.get_activation("blue car")
        blue_activation = self.model.get_activation("blue")
        car_activation = self.model.get_activation("car")
        avg_activation = (blue_activation + car_activation) / 2.0
        sim = cosine_similarity(phrase_activation.unsqueeze(0).numpy(), avg_activation.unsqueeze(0).numpy())[0][0]
        self.assertGreater(sim, 0.7)

    def test_feature_interpretability(self):
        tokens = ['solve', 'reason', 'conclude', 'analyze', 'compute', 'apple', 'orange', 'banana', 'car', 'truck',
                  'bus']
        fractions = []
        for token in tokens:
            activation = self.model.get_activation(token)
            fraction = (activation > 0).sum().item() / self.model.num_atoms
            fractions.append(fraction)
        avg_fraction = np.mean(fractions)
        self.assertGreater(avg_fraction, 0.01)
        self.assertLess(avg_fraction, 0.5)

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
                avg_sim = np.mean(sim_matrix[np.triu_indices(len(toks), k=1)])
                self.assertGreater(avg_sim, 0.2)


if __name__ == '__main__':
    unittest.main()