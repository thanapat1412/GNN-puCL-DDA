"""
utils.py — Shared Utilities for Drug-Disease Association Prediction

This module provides shared utility functions used across all experimental
settings, including graph construction, embedding generation, feature
engineering, downstream PU-bagging classification, and dataset builders.

Reference:
    T. Sinsrangboon and T. Panitanarak, "A Linearly Scalable GNN
    Framework on Drug-Gene Ontology-Disease Tripartite Graphs for
    Drug-Disease Association Prediction With Positive-Unlabeled
    Contrastive Learning," IEEE Access, 2026.
"""

import random
import time

import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve, accuracy_score
)
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import HeteroData
import xgboost as xgb


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed=42):
    """
    Set random seeds across all libraries for reproducibility.

    Args:
        seed (int): Random seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Graph Construction
# ============================================================================

def create_heterogeneous_graph(
    drug_go_matrix,
    disease_go_matrix,
    drug_disease_matrix,
    num_drugs,
    num_diseases,
    num_go_terms,
    use_one_hot_features=True
):
    """
    Construct a heterogeneous tripartite graph from adjacency matrices.

    The graph contains three node types (drug, disease, GO) and three
    bidirectional edge types (drug-GO, disease-GO, drug-disease).

    Args:
        drug_go_matrix (np.ndarray): Binary adjacency matrix (num_drugs x num_GO).
        disease_go_matrix (np.ndarray): Binary adjacency matrix (num_diseases x num_GO).
        drug_disease_matrix (np.ndarray): Binary adjacency matrix (num_drugs x num_diseases).
        num_drugs (int): Number of drug nodes.
        num_diseases (int): Number of disease nodes.
        num_go_terms (int): Number of GO term nodes.
        use_one_hot_features (bool): If True, initialize nodes with one-hot identity
            vectors (for Linear encoders). If False, use integer indices
            (for Embedding encoders). Default: True.

    Returns:
        HeteroData: PyTorch Geometric heterogeneous graph object.
    """
    hetero_graph = HeteroData()

    # Initialize node features
    if use_one_hot_features:
        hetero_graph['drug'].x = torch.eye(num_drugs, dtype=torch.float)
        hetero_graph['disease'].x = torch.eye(num_diseases, dtype=torch.float)
        hetero_graph['go'].x = torch.eye(num_go_terms, dtype=torch.float)
    else:
        hetero_graph['drug'].x = torch.arange(num_drugs, dtype=torch.long).unsqueeze(1)
        hetero_graph['disease'].x = torch.arange(num_diseases, dtype=torch.long).unsqueeze(1)
        hetero_graph['go'].x = torch.arange(num_go_terms, dtype=torch.long).unsqueeze(1)

    # Drug-GO edges (bidirectional)
    drug_idx, go_idx = np.where(drug_go_matrix)
    hetero_graph['drug', 'connects', 'go'].edge_index = torch.tensor(
        [drug_idx, go_idx], dtype=torch.long
    )
    hetero_graph['go', 'rev_connects', 'drug'].edge_index = torch.tensor(
        [go_idx, drug_idx], dtype=torch.long
    )

    # Disease-GO edges (bidirectional)
    dis_idx, go_idx = np.where(disease_go_matrix)
    hetero_graph['disease', 'connects', 'go'].edge_index = torch.tensor(
        [dis_idx, go_idx], dtype=torch.long
    )
    hetero_graph['go', 'rev_connects', 'disease'].edge_index = torch.tensor(
        [go_idx, dis_idx], dtype=torch.long
    )

    # Drug-disease edges (bidirectional)
    drug_idx, dis_idx = np.where(drug_disease_matrix)
    hetero_graph['drug', 'connects', 'disease'].edge_index = torch.tensor(
        [drug_idx, dis_idx], dtype=torch.long
    )
    hetero_graph['disease', 'rev_connects', 'drug'].edge_index = torch.tensor(
        [dis_idx, drug_idx], dtype=torch.long
    )

    return hetero_graph


# ============================================================================
# Embedding Generation
# ============================================================================

def generate_embeddings(model, hetero_graph, device='cuda', training=True):
    """
    Generate node embeddings from the heterogeneous GNN.

    Args:
        model (HeteroGNN): The GNN model instance.
        hetero_graph (HeteroData): The heterogeneous graph.
        device (str): Computation device ('cuda' or 'cpu').
        training (bool): If True, enable gradient computation for
            backpropagation. If False, run in inference mode.

    Returns:
        tuple: (embeddings_dict, elapsed_time)
            - embeddings_dict (dict): Node embeddings keyed by node type.
            - elapsed_time (float): Time in seconds for forward pass.
    """
    hetero_graph = hetero_graph.to(device)
    model = model.to(device)

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    start_time = time.time()
    embeddings = model(hetero_graph.x_dict, hetero_graph.edge_index_dict)
    elapsed_time = time.time() - start_time

    return embeddings, elapsed_time


# ============================================================================
# Dataset Builders for Downstream Classification
# ============================================================================

def build_training_set(
    train_drug_disease_matrix,
    drug_embeddings,
    disease_embeddings,
    drug_features=None,
    disease_features=None
):
    """
    Build training set for the PU-bagging classifier from GNN embeddings.

    Each drug-disease pair is represented by concatenating the drug and
    disease embeddings (and optionally engineered features). Labels are
    1 for known positive associations and 0 for unlabeled pairs.

    Args:
        train_drug_disease_matrix (np.ndarray): Binary matrix (num_drugs x num_diseases),
            where 1 = known association, 0 = unlabeled.
        drug_embeddings (Tensor): Drug embeddings (num_drugs, dim).
        disease_embeddings (Tensor): Disease embeddings (num_diseases, dim).
        drug_features (np.ndarray, optional): Engineered drug features.
        disease_features (np.ndarray, optional): Engineered disease features.

    Returns:
        tuple: (X_train, y_train) as numpy arrays.
    """
    X_train = []
    y_train = []
    num_drugs, num_diseases = train_drug_disease_matrix.shape

    for i in range(num_drugs):
        for j in range(num_diseases):
            label = train_drug_disease_matrix[i, j]
            drug_emb = drug_embeddings[i].detach().cpu().numpy()
            disease_emb = disease_embeddings[j].detach().cpu().numpy()

            combined = [drug_emb, disease_emb]
            if drug_features is not None and disease_features is not None:
                combined.append(drug_features[i])
                combined.append(disease_features[j])
            feature_vector = np.concatenate(combined, axis=0)

            X_train.append(feature_vector)
            y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train).astype(int)

    return X_train, y_train


def build_test_set(
    test_edges,
    drug_embeddings,
    disease_embeddings,
    drug_disease_matrix,
    drug_features=None,
    disease_features=None,
    seed=42
):
    """
    Build test set with balanced positive and unlabeled samples.

    Positive samples are the held-out test edges. An equal number of
    unlabeled pairs (not in the full association matrix or test set)
    are randomly sampled as negative examples.

    Args:
        test_edges (np.ndarray): Held-out positive edges, shape (N, 2).
        drug_embeddings (Tensor): Drug embeddings (num_drugs, dim).
        disease_embeddings (Tensor): Disease embeddings (num_diseases, dim).
        drug_disease_matrix (np.ndarray): Full binary association matrix.
        drug_features (np.ndarray, optional): Engineered drug features.
        disease_features (np.ndarray, optional): Engineered disease features.
        seed (int): Random seed for negative sampling (default: 42).

    Returns:
        tuple: (X_test, y_test) as numpy arrays.
    """
    random.seed(seed)
    np.random.seed(seed)

    num_drugs, num_diseases = drug_disease_matrix.shape

    # Positive samples from held-out test edges
    pos_features = []
    for i, j in test_edges:
        drug_emb = drug_embeddings[i].detach().cpu().numpy()
        disease_emb = disease_embeddings[j].detach().cpu().numpy()

        combined = [drug_emb, disease_emb]
        if drug_features is not None and disease_features is not None:
            combined.append(drug_features[i])
            combined.append(disease_features[j])
        pos_features.append(np.concatenate(combined, axis=0))

    pos_labels = [1] * len(pos_features)

    # Negative samples from unlabeled pairs
    pos_set = set((i, j) for i, j in test_edges)
    all_unlabeled = [
        (i, j) for i in range(num_drugs)
        for j in range(num_diseases)
        if drug_disease_matrix[i, j] == 0 and (i, j) not in pos_set
    ]
    sampled_unlabeled = random.sample(all_unlabeled, len(test_edges))

    neg_features = []
    for i, j in sampled_unlabeled:
        drug_emb = drug_embeddings[i].detach().cpu().numpy()
        disease_emb = disease_embeddings[j].detach().cpu().numpy()

        combined = [drug_emb, disease_emb]
        if drug_features is not None and disease_features is not None:
            combined.append(drug_features[i])
            combined.append(disease_features[j])
        neg_features.append(np.concatenate(combined, axis=0))

    neg_labels = [0] * len(neg_features)

    X_test = np.array(pos_features + neg_features)
    y_test = np.array(pos_labels + neg_labels)

    return X_test, y_test


# ============================================================================
# PU-Bagging Classifier
# ============================================================================

def run_pu_bagging_classifier(X_train, y_train, X_test, y_test,
                               T=50, random_state=42, xgb_params=None):
    """
    Train and evaluate a PU-bagging ensemble classifier.

    In each of T iterations, a balanced subset is created by pairing all
    positive samples with an equal-sized bootstrap sample from the unlabeled
    pool. An XGBoost classifier is trained on each subset, and predictions
    are averaged across all iterations.

    The optimal decision threshold is selected by maximizing F1_PU, a metric
    designed for the positive-unlabeled setting:
        F1_PU = (Recall^2 * N) / (TP + UP)
    where N is the total test samples, TP is true positives, and UP is
    unlabeled samples predicted as positive.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels (1 = positive, 0 = unlabeled).
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): Test labels.
        T (int): Number of bagging iterations (default: 50).
        random_state (int): Base random seed (default: 42).
        xgb_params (dict, optional): XGBoost hyperparameters.

    Returns:
        tuple: (metrics_dict, prob_scores)
            - metrics_dict (dict): Evaluation metrics (AUPRC, AUROC, Precision,
              Recall, Accuracy, F1, F1_PU, Threshold).
            - prob_scores (np.ndarray): Averaged prediction probabilities.
    """
    np.random.seed(random_state)
    random.seed(random_state)

    if xgb_params is None:
        xgb_params = {
            'learning_rate': 0.1,
            'n_estimators': 200,
            'max_depth': 9,
            'min_child_weight': 1,
            'eval_metric': 'logloss',
            'device': 'cuda'
        }

    pos_idx = np.where(y_train == 1)[0]
    unlabeled_idx = np.where(y_train == 0)[0]
    prob_scores = np.zeros(len(X_test))

    # Bagging iterations
    for t in range(T):
        sampled_unlabeled = np.random.choice(unlabeled_idx, size=len(pos_idx), replace=True)
        train_idx = np.concatenate([pos_idx, sampled_unlabeled])
        train_X = X_train[train_idx]
        train_y = np.array([1] * len(pos_idx) + [0] * len(sampled_unlabeled))

        clf = xgb.XGBClassifier(random_state=t, **xgb_params)
        clf.fit(train_X, train_y)
        prob_scores += clf.predict_proba(X_test)[:, 1]

    prob_scores /= T

    # Compute F1_PU across all thresholds to find optimal decision boundary
    N = len(y_test)
    precision, recall, thresholds = precision_recall_curve(y_test, prob_scores)

    f1_pu_scores = []
    for i, th in enumerate(thresholds):
        y_pred_at_th = (prob_scores >= th).astype(int)
        TP = np.sum((y_test == 1) & (y_pred_at_th == 1))
        UP = np.sum((y_test == 0) & (y_pred_at_th == 1))
        denominator = TP + UP
        if denominator > 0:
            f1_pu_val = (recall[i] ** 2 * N) / denominator
        else:
            f1_pu_val = 0.0
        f1_pu_scores.append(f1_pu_val)

    f1_pu_scores = np.array(f1_pu_scores)
    max_idx = np.argmax(f1_pu_scores)
    threshold_score = thresholds[max_idx]
    best_f1_pu = f1_pu_scores[max_idx]

    # Compute all metrics at the optimal threshold
    y_pred = (prob_scores >= threshold_score).astype(int)

    metrics_dict = {
        'AUPRC': average_precision_score(y_test, prob_scores),
        'AUROC': roc_auc_score(y_test, prob_scores),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'F1_PU': best_f1_pu,
        'Threshold': threshold_score
    }

    return metrics_dict, prob_scores
