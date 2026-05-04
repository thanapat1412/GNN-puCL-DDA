"""
run_setting3.py — Experimental Settings 1-3

This script implements the first three experimental settings for
drug-disease association prediction on the tripartite graph:

  Setting 1: Structure-only propagation (untrained GNN, single forward pass).
  Setting 2: Supervised training with binary cross-entropy (BCE) loss.
  Setting 3: Setting 2 + positive-unlabeled contrastive learning (puCL).

All three settings share the same GNN architecture, graph construction,
and PU-bagging evaluation pipeline. The difference lies in the training
objective:
  - Setting 1 uses random weights with no parameter updates.
  - Setting 2 trains with BCE loss only.
  - Setting 3 trains with BCE + puCL contrastive loss (Eq. 3 in the paper).

Usage:
    python run_setting3.py --setting 3 --phase 2

    Adjust DATA_DIR and OUTPUT_DIR below to point to your local data files.
    See README.md for data preparation instructions.

Reference:
    T. Sinsrangboon and T. Panitanarak, "A Linearly Scalable GNN
    Framework on Drug-Gene Ontology-Disease Tripartite Graphs for
    Drug-Disease Association Prediction With Positive-Unlabeled
    Contrastive Learning," IEEE Access, 2026.
"""

import os
import argparse
import pickle
import time

import numpy as np
import pandas as pd
import torch

from models import HeteroGNN, compute_bce_loss, compute_pucl_contrastive_loss
from utils import (
    set_seed,
    create_heterogeneous_graph,
    generate_embeddings,
    build_training_set,
    build_test_set,
    run_pu_bagging_classifier
)


# ============================================================================
# Configuration
# ============================================================================

# Data directory — users must place data files here (see README.md)
DATA_DIR = "./data"

# Output directory for embeddings and results
OUTPUT_DIR = "./outputs"

# Hyperparameters (consistent across all settings)
HIDDEN_CHANNELS = 64
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PU_BAGGING_ITERATIONS = 50
CONTRASTIVE_WEIGHT = 1.0   # lambda in Eq. 3
CONTRASTIVE_TEMP = 0.1     # tau in Eq. 2


# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_dir, phase=2):
    """
    Load adjacency matrices, node lists, and cross-validation folds.

    Expected files in data_dir (for the specified phase):
      - drug_go_matrix_Phase{phase}.txt
      - disease_go_matrix_Phase{phase}.txt
      - drug_disease_matrix_Phase{phase}.txt
      - drug_list_Phase{phase}.csv
      - disease_list_Phase{phase}.csv
      - go_union_list_Phase{phase}.csv
      - folds_5folds_Phase{phase}.pkl

    Args:
        data_dir (str): Path to the data directory.
        phase (int): Dataset phase (1 = 2019 data, 2 = 2025 data).

    Returns:
        dict: Dictionary containing all loaded data objects.
    """
    suffix = f"Phase{phase}"

    # Load adjacency matrices
    drug_go_matrix = np.loadtxt(
        os.path.join(data_dir, f"drug_go_matrix_{suffix}.txt"),
        delimiter='\t', dtype=np.int8
    )
    disease_go_matrix = np.loadtxt(
        os.path.join(data_dir, f"disease_go_matrix_{suffix}.txt"),
        delimiter='\t', dtype=np.int8
    )
    drug_disease_matrix = np.loadtxt(
        os.path.join(data_dir, f"drug_disease_matrix_{suffix}.txt"),
        delimiter='\t', dtype=np.int8
    )

    # Load node lists
    drug_list = pd.read_csv(
        os.path.join(data_dir, f"drug_list_{suffix}.csv")
    )['drugBankId'].reset_index(drop=True)
    disease_list = pd.read_csv(
        os.path.join(data_dir, f"disease_list_{suffix}.csv")
    )['diseaseUMLSCUI'].reset_index(drop=True)
    go_list = pd.read_csv(
        os.path.join(data_dir, f"go_union_list_{suffix}.csv")
    )['goId'].reset_index(drop=True)

    # Load cross-validation folds
    with open(os.path.join(data_dir, f"folds_5folds_{suffix}.pkl"), 'rb') as f:
        folds = pickle.load(f)

    # Print network statistics
    print(f"Network Statistics (Phase {phase}):")
    print(f"  Drugs: {len(drug_list)}")
    print(f"  Diseases: {len(disease_list)}")
    print(f"  GO terms: {len(go_list)}")
    print(f"  Drug-GO edges: {drug_go_matrix.sum()}")
    print(f"  Disease-GO edges: {disease_go_matrix.sum()}")
    print(f"  Drug-disease associations: {drug_disease_matrix.sum()}")
    print(f"  Cross-validation folds: {len(folds)}")

    return {
        'drug_go_matrix': drug_go_matrix,
        'disease_go_matrix': disease_go_matrix,
        'drug_disease_matrix': drug_disease_matrix,
        'drug_list': drug_list,
        'disease_list': disease_list,
        'go_list': go_list,
        'folds': folds
    }


# ============================================================================
# Training Functions
# ============================================================================

def run_setting1(model, hetero_graph, device):
    """
    Setting 1: Structure-only propagation (no training).

    The GNN is initialized with random weights and applied in a single
    forward pass. No parameters are updated. This tests whether the
    tripartite network topology alone carries a predictive signal.

    Args:
        model (HeteroGNN): Initialized GNN model (random weights).
        hetero_graph (HeteroData): The heterogeneous graph.
        device (str): Computation device.

    Returns:
        tuple: (drug_emb, disease_emb, go_emb, embedding_time)
    """
    embeddings, emb_time = generate_embeddings(
        model, hetero_graph, device=device, training=False
    )
    return embeddings['drug'], embeddings['disease'], embeddings['go'], emb_time


def run_setting2(model, hetero_graph, positive_pairs, device):
    """
    Setting 2: Supervised training with BCE loss.

    The GNN is trained for NUM_EPOCHS epochs using binary cross-entropy
    loss on known positive associations and sampled unlabeled pairs.

    Args:
        model (HeteroGNN): Initialized GNN model.
        hetero_graph (HeteroData): The heterogeneous graph.
        positive_pairs (np.ndarray): Known positive drug-disease pairs.
        device (str): Computation device.

    Returns:
        tuple: (drug_emb, disease_emb, go_emb, total_embedding_time)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    emb_times = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()

        embeddings, emb_time = generate_embeddings(
            model, hetero_graph, device=device, training=True
        )
        emb_times.append(emb_time)

        drug_emb = embeddings['drug']
        disease_emb = embeddings['disease']

        loss = compute_bce_loss(
            drug_emb, disease_emb, positive_pairs,
            num_neg_samples=len(positive_pairs)
        )

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}: BCE Loss = {loss.item():.4f}")

    return drug_emb, disease_emb, embeddings['go'], sum(emb_times)


def run_setting3(model, hetero_graph, positive_pairs, device):
    """
    Setting 3: BCE + puCL contrastive learning.

    The GNN is trained with the combined objective (Eq. 3):
        L = L_BCE + lambda * L_contrastive
    where lambda = CONTRASTIVE_WEIGHT.

    Args:
        model (HeteroGNN): Initialized GNN model.
        hetero_graph (HeteroData): The heterogeneous graph.
        positive_pairs (np.ndarray): Known positive drug-disease pairs.
        device (str): Computation device.

    Returns:
        tuple: (drug_emb, disease_emb, go_emb, total_embedding_time)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    emb_times = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()

        embeddings, emb_time = generate_embeddings(
            model, hetero_graph, device=device, training=True
        )
        emb_times.append(emb_time)

        drug_emb = embeddings['drug']
        disease_emb = embeddings['disease']

        # BCE loss component (Eq. 1)
        bce_loss = compute_bce_loss(
            drug_emb, disease_emb, positive_pairs,
            num_neg_samples=len(positive_pairs)
        )

        # puCL contrastive loss component (Eq. 2)
        contrastive_loss = compute_pucl_contrastive_loss(
            drug_emb, disease_emb, positive_pairs,
            temperature=CONTRASTIVE_TEMP
        )

        # Combined objective (Eq. 3)
        total_loss = bce_loss + CONTRASTIVE_WEIGHT * contrastive_loss
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}: Total = {total_loss.item():.4f} "
                  f"(BCE = {bce_loss.item():.4f}, "
                  f"Contrastive = {contrastive_loss.item():.4f})")

    return drug_emb, disease_emb, embeddings['go'], sum(emb_times)


# ============================================================================
# Main Execution
# ============================================================================

def main(setting, phase):
    """
    Run the specified experimental setting with 5-fold cross-validation.

    Args:
        setting (int): Experimental setting (1, 2, or 3).
        phase (int): Dataset phase (1 or 2).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Running Setting {setting} on Phase {phase} data\n")

    # Load data
    data = load_data(DATA_DIR, phase=phase)
    drug_disease_matrix = data['drug_disease_matrix']
    folds = data['folds']
    num_drugs = len(data['drug_list'])
    num_diseases = len(data['disease_list'])
    num_go_terms = len(data['go_list'])

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = []

    for fold in folds:
        fold_id = fold['fold_id']
        set_seed(42 + fold_id)

        train_edges = np.array(fold['train_edges'])
        test_edges = np.array(fold['test_edges'])

        print(f"{'='*60}")
        print(f"Fold {fold_id}: Train edges = {len(train_edges)}, "
              f"Test edges = {len(test_edges)}")
        print(f"{'='*60}")

        start_time = time.time()

        # Reset GPU memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # Mask test edges from the training graph to prevent information leakage
        train_drug_disease_matrix = drug_disease_matrix.copy()
        for i, j in test_edges:
            train_drug_disease_matrix[i, j] = 0

        # Construct heterogeneous graph from training data
        use_one_hot = (setting in [1, 2, 3])  # Settings 1-3 use one-hot features
        hetero_graph = create_heterogeneous_graph(
            drug_go_matrix=data['drug_go_matrix'],
            disease_go_matrix=data['disease_go_matrix'],
            drug_disease_matrix=train_drug_disease_matrix,
            num_drugs=num_drugs,
            num_diseases=num_diseases,
            num_go_terms=num_go_terms,
            use_one_hot_features=use_one_hot
        )

        # Initialize GNN model
        model = HeteroGNN(
            num_drugs=num_drugs,
            num_diseases=num_diseases,
            num_go_terms=num_go_terms,
            hidden_channels=HIDDEN_CHANNELS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            use_learnable_embeddings=False
        ).to(device)

        # Run the selected setting
        positive_pairs = np.argwhere(train_drug_disease_matrix == 1)

        if setting == 1:
            drug_emb, disease_emb, go_emb, emb_time = run_setting1(
                model, hetero_graph, device
            )
        elif setting == 2:
            drug_emb, disease_emb, go_emb, emb_time = run_setting2(
                model, hetero_graph, positive_pairs, device
            )
        elif setting == 3:
            drug_emb, disease_emb, go_emb, emb_time = run_setting3(
                model, hetero_graph, positive_pairs, device
            )
        else:
            raise ValueError(f"Invalid setting: {setting}. Choose 1, 2, or 3.")

        train_end = time.time()

        # Save embeddings (needed by Setting 4 and Top-K analysis)
        if setting == 3:
            emb_dir = os.path.join(OUTPUT_DIR, "embeddings")
            os.makedirs(emb_dir, exist_ok=True)
            np.save(
                os.path.join(emb_dir, f"drug_emb_setting3_fold{fold_id}.npy"),
                drug_emb.detach().cpu().numpy()
            )
            np.save(
                os.path.join(emb_dir, f"disease_emb_setting3_fold{fold_id}.npy"),
                disease_emb.detach().cpu().numpy()
            )
            np.save(
                os.path.join(emb_dir, f"go_emb_setting3_fold{fold_id}.npy"),
                go_emb.detach().cpu().numpy()
            )
            print(f"  Embeddings saved to {emb_dir}")

        # ---- Downstream evaluation with PU-bagging ----
        downstream_start = time.time()

        X_train, y_train = build_training_set(
            train_drug_disease_matrix=train_drug_disease_matrix,
            drug_embeddings=drug_emb,
            disease_embeddings=disease_emb
        )
        X_test, y_test = build_test_set(
            test_edges=test_edges,
            drug_embeddings=drug_emb,
            disease_embeddings=disease_emb,
            drug_disease_matrix=drug_disease_matrix
        )

        metrics, prob_scores = run_pu_bagging_classifier(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            T=PU_BAGGING_ITERATIONS
        )

        downstream_end = time.time()
        end_time = time.time()

        # Record timing and memory metrics
        metrics['fold_id'] = fold_id
        metrics['total_sec'] = end_time - start_time
        metrics['train_sec'] = train_end - start_time
        metrics['downstream_sec'] = downstream_end - downstream_start

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            max_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
            metrics['max_gpu_mem_mb'] = max_mem

        print(f"\n  Fold {fold_id} Results:")
        for key in ['AUPRC', 'AUROC', 'Precision', 'Recall', 'Accuracy',
                     'F1', 'F1_PU']:
            print(f"    {key:12s}: {metrics[key]:.4f}")
        print(f"    Time: {metrics['total_sec']:.1f}s")
        print()

        all_metrics.append(metrics)

    # ---- Summary across all folds ----
    print(f"\n{'='*60}")
    print(f"Summary: Setting {setting}, Phase {phase} (5-fold CV)")
    print(f"{'='*60}")

    for key in ['AUPRC', 'AUROC', 'Precision', 'Recall', 'Accuracy',
                 'F1', 'F1_PU']:
        vals = [m[key] for m in all_metrics]
        print(f"  {key:12s}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    if 'max_gpu_mem_mb' in all_metrics[0]:
        mem_vals = [m['max_gpu_mem_mb'] for m in all_metrics]
        print(f"  {'GPU Memory':12s}: {np.mean(mem_vals):.0f} +/- {np.std(mem_vals):.0f} MB")

    time_vals = [m['total_sec'] for m in all_metrics]
    print(f"  {'Runtime':12s}: {np.mean(time_vals):.1f} +/- {np.std(time_vals):.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Settings 1-3 for drug-disease association prediction."
    )
    parser.add_argument(
        "--setting", type=int, default=3, choices=[1, 2, 3],
        help="Experimental setting to run (1, 2, or 3). Default: 3."
    )
    parser.add_argument(
        "--phase", type=int, default=2, choices=[1, 2],
        help="Dataset phase (1 = 2019 data, 2 = 2025 data). Default: 2."
    )
    args = parser.parse_args()

    main(setting=args.setting, phase=args.phase)
