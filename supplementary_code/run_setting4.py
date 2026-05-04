"""
run_setting4.py — Experimental Setting 4: Coverage-Aware Semantic Integration

This script implements Setting 4, which extends Setting 3 by fusing
structural GNN embeddings with semantic features from biomedical language
models (PubMedBERT or BioBERT).

The pipeline consists of:
  1. Load pre-computed Setting 3 embeddings (from run_setting3.py).
  2. Load BERT embeddings extracted from drug mechanism descriptions.
  3. Apply PCA to compress BERT embeddings from 128d (pre-compressed from 768d) to 64d.
  4. Apply MinMax scaling to both GNN and BERT embeddings.
  5. Apply coverage-aware fusion (Eq. 4 in the paper):
     - Drugs WITH descriptions: fused = p * GNN + (1-p) * BERT
     - Drugs WITHOUT descriptions: fused = GNN (100%)
  6. Initialize GNN encoder weights with fused embeddings.
  7. Retrain for 50 epochs with the combined objective (Eq. 3).
  8. Evaluate with PU-bagging classifier.

Usage:
    python run_setting4.py --phase 2

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

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

# Output directory (must contain embeddings from run_setting3.py)
OUTPUT_DIR = "./outputs"

# BERT embedding files (pre-extracted from 768d and pre-compressed to 128d)
BERT_EMBEDDING_FILES = {
    'PubMedBERT': 'drug_pubmedbert_embeddings_128d.npy',
    'BioBERT': 'drug_biobert_embeddings_128d.npy'
}

# Fusion weights to evaluate (p = GNN weight, 1-p = BERT weight)
P_VALUES = [0.3, 0.5, 0.7]

# PCA target dimensionality (must match GNN hidden_channels)
PCA_TARGET_DIM = 64

# Hyperparameters (same as Setting 3)
HIDDEN_CHANNELS = 64
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PU_BAGGING_ITERATIONS = 50
CONTRASTIVE_WEIGHT = 1.0
CONTRASTIVE_TEMP = 0.1


# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_dir, phase=2):
    """
    Load adjacency matrices, node lists, and cross-validation folds.

    Args:
        data_dir (str): Path to the data directory.
        phase (int): Dataset phase (1 or 2).

    Returns:
        dict: Dictionary containing all loaded data objects.
    """
    suffix = f"Phase{phase}"

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

    drug_list = pd.read_csv(
        os.path.join(data_dir, f"drug_list_{suffix}.csv")
    )['drugBankId'].reset_index(drop=True)
    disease_list = pd.read_csv(
        os.path.join(data_dir, f"disease_list_{suffix}.csv")
    )['diseaseUMLSCUI'].reset_index(drop=True)
    go_list = pd.read_csv(
        os.path.join(data_dir, f"go_union_list_{suffix}.csv")
    )['goId'].reset_index(drop=True)

    with open(os.path.join(data_dir, f"folds_5folds_{suffix}.pkl"), 'rb') as f:
        folds = pickle.load(f)

    print(f"Network Statistics (Phase {phase}):")
    print(f"  Drugs: {len(drug_list)}")
    print(f"  Diseases: {len(disease_list)}")
    print(f"  GO terms: {len(go_list)}")
    print(f"  Drug-disease associations: {drug_disease_matrix.sum()}")

    return {
        'drug_go_matrix': drug_go_matrix,
        'disease_go_matrix': disease_go_matrix,
        'drug_disease_matrix': drug_disease_matrix,
        'drug_list': drug_list,
        'disease_list': disease_list,
        'go_list': go_list,
        'folds': folds
    }


def load_setting3_embeddings(output_dir, num_folds=5):
    """
    Load pre-computed Setting 3 embeddings for all folds.

    These embeddings are produced by run_setting3.py with --setting 3
    and saved in the embeddings subdirectory.

    Args:
        output_dir (str): Base output directory.
        num_folds (int): Number of cross-validation folds.

    Returns:
        dict: Dictionaries mapping fold_id to numpy arrays for drug,
              disease, and GO embeddings.
    """
    emb_dir = os.path.join(output_dir, "embeddings")
    drug_embs, disease_embs, go_embs = {}, {}, {}

    for fold_id in range(num_folds):
        drug_embs[fold_id] = np.load(
            os.path.join(emb_dir, f"drug_emb_setting3_fold{fold_id}.npy")
        )
        disease_embs[fold_id] = np.load(
            os.path.join(emb_dir, f"disease_emb_setting3_fold{fold_id}.npy")
        )
        go_embs[fold_id] = np.load(
            os.path.join(emb_dir, f"go_emb_setting3_fold{fold_id}.npy")
        )
        print(f"  Fold {fold_id}: Drug {drug_embs[fold_id].shape}, "
              f"Disease {disease_embs[fold_id].shape}, "
              f"GO {go_embs[fold_id].shape}")

    return drug_embs, disease_embs, go_embs


def load_bert_embeddings(data_dir):
    """
    Load pre-extracted BERT embeddings for drug descriptions.

    Drugs without textual descriptions have all-zero embedding vectors.
    The coverage rate is printed for each model.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        dict: BERT embeddings keyed by model name.
    """
    bert_embeddings = {}

    for model_name, filename in BERT_EMBEDDING_FILES.items():
        filepath = os.path.join(data_dir, filename)
        emb = np.load(filepath)

        coverage = np.sum(~np.all(emb == 0, axis=1))
        total = len(emb)
        print(f"  {model_name}: {emb.shape}, "
              f"Coverage: {coverage}/{total} ({coverage/total*100:.1f}%)")

        bert_embeddings[model_name] = emb

    return bert_embeddings


# ============================================================================
# BERT Preprocessing and Fusion
# ============================================================================

def apply_pca_to_bert(bert_embeddings, target_dim=64):
    """
    Reduce BERT embedding dimensionality using PCA.

    PCA is fitted only on drugs with non-zero embeddings (i.e., drugs
    that have textual descriptions). Zero-vector drugs remain unchanged.

    Args:
        bert_embeddings (dict): BERT embeddings keyed by model name.
        target_dim (int): Target dimensionality (default: 64).

    Returns:
        dict: Reduced BERT embeddings keyed by model name.
    """
    pca = PCA(n_components=target_dim, random_state=42)
    reduced = {}

    for model_name, emb in bert_embeddings.items():
        non_zero_mask = ~np.all(emb == 0, axis=1)

        pca.fit(emb[non_zero_mask])
        variance_explained = pca.explained_variance_ratio_.sum()

        emb_reduced = np.zeros((emb.shape[0], target_dim))
        emb_reduced[non_zero_mask] = pca.transform(emb[non_zero_mask])

        reduced[model_name] = emb_reduced
        print(f"  {model_name}: {emb.shape[1]}d -> {target_dim}d, "
              f"Variance retained: {variance_explained*100:.2f}%")

    return reduced


def scale_embeddings(embeddings):
    """
    Apply MinMax scaling to [0, 1] range.

    For BERT embeddings, only non-zero rows are scaled to preserve
    the zero-vector convention for drugs without descriptions.

    Args:
        embeddings (np.ndarray): Input embeddings.

    Returns:
        np.ndarray: Scaled embeddings.
    """
    non_zero_mask = ~np.all(embeddings == 0, axis=1)

    if non_zero_mask.all():
        # All rows are non-zero: scale everything
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(embeddings)
    else:
        # Only scale non-zero rows; preserve zero rows
        scaled = np.zeros_like(embeddings)
        if non_zero_mask.sum() > 0:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled[non_zero_mask] = scaler.fit_transform(embeddings[non_zero_mask])
        return scaled


def create_fused_drug_embeddings(drug_gnn_scaled, bert_scaled, p):
    """
    Apply coverage-aware fusion (Eq. 4 in the paper).

    For drugs with BERT embeddings (non-zero rows):
        fused = p * GNN_scaled + (1 - p) * BERT_scaled
    For drugs without BERT embeddings (zero rows):
        fused = GNN_scaled  (100% structural)

    This ensures that drugs lacking textual descriptions are not
    penalized by the introduction of artificial zero vectors.

    Args:
        drug_gnn_scaled (np.ndarray): Scaled GNN embeddings (num_drugs, dim).
        bert_scaled (np.ndarray): Scaled BERT embeddings (num_drugs, dim).
        p (float): Fusion weight for GNN component, in [0, 1].

    Returns:
        np.ndarray: Fused drug embeddings (num_drugs, dim).
    """
    assert drug_gnn_scaled.shape == bert_scaled.shape, "Shape mismatch."
    assert 0 <= p <= 1, "p must be in [0, 1]."

    has_bert = ~np.all(bert_scaled == 0, axis=1)

    # Default: use GNN embeddings for all drugs
    drug_fused = drug_gnn_scaled.copy()

    # Apply weighted fusion only for drugs with BERT coverage
    if has_bert.sum() > 0:
        drug_fused[has_bert] = (
            p * drug_gnn_scaled[has_bert] + (1 - p) * bert_scaled[has_bert]
        )

    return drug_fused


# ============================================================================
# Main Execution
# ============================================================================

def main(phase):
    """
    Run Setting 4 across all BERT models, fusion weights, and folds.

    Args:
        phase (int): Dataset phase (1 or 2).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Running Setting 4 on Phase {phase} data\n")

    # Load data
    data = load_data(DATA_DIR, phase=phase)
    drug_disease_matrix = data['drug_disease_matrix']
    folds = data['folds']
    num_drugs = len(data['drug_list'])
    num_diseases = len(data['disease_list'])
    num_go_terms = len(data['go_list'])

    # Load pre-computed embeddings
    print("\nLoading Setting 3 embeddings...")
    drug_s3_embs, disease_s3_embs, go_s3_embs = load_setting3_embeddings(OUTPUT_DIR)

    print("\nLoading BERT embeddings...")
    bert_embeddings_raw = load_bert_embeddings(DATA_DIR)

    # Preprocessing: PCA reduction
    print("\nApplying PCA to BERT embeddings...")
    bert_64d = apply_pca_to_bert(bert_embeddings_raw, target_dim=PCA_TARGET_DIM)

    # Preprocessing: MinMax scaling for BERT
    print("\nScaling BERT embeddings...")
    bert_scaled = {}
    for model_name, emb in bert_64d.items():
        bert_scaled[model_name] = scale_embeddings(emb)

    # Preprocessing: MinMax scaling for Setting 3 embeddings (per fold)
    print("\nScaling Setting 3 embeddings...")
    drug_s3_scaled = {}
    disease_s3_scaled = {}
    go_s3_scaled = {}
    for fold_id in range(len(folds)):
        drug_s3_scaled[fold_id] = scale_embeddings(drug_s3_embs[fold_id])
        disease_s3_scaled[fold_id] = scale_embeddings(disease_s3_embs[fold_id])
        go_s3_scaled[fold_id] = scale_embeddings(go_s3_embs[fold_id])

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Main experiment loop ----
    total_configs = len(BERT_EMBEDDING_FILES) * len(P_VALUES) * len(folds)
    print(f"\nTotal configurations: {total_configs} "
          f"({len(BERT_EMBEDDING_FILES)} BERT models x "
          f"{len(P_VALUES)} p-values x {len(folds)} folds)")

    all_results = []
    config_counter = 0

    for fold_idx, fold in enumerate(folds):
        fold_id = fold['fold_id']
        train_edges = np.array(fold['train_edges'])
        test_edges = np.array(fold['test_edges'])

        set_seed(42 + fold_id)

        # Mask test edges for information leakage prevention
        train_drug_disease_matrix = drug_disease_matrix.copy()
        for i, j in test_edges:
            train_drug_disease_matrix[i, j] = 0

        # Construct heterogeneous graph (shared across BERT models and p-values)
        hetero_graph = create_heterogeneous_graph(
            drug_go_matrix=data['drug_go_matrix'],
            disease_go_matrix=data['disease_go_matrix'],
            drug_disease_matrix=train_drug_disease_matrix,
            num_drugs=num_drugs,
            num_diseases=num_diseases,
            num_go_terms=num_go_terms,
            use_one_hot_features=False  # Setting 4 uses Embedding encoders
        )

        positive_pairs = np.argwhere(train_drug_disease_matrix == 1)

        for bert_name in BERT_EMBEDDING_FILES:
            for p in P_VALUES:
                config_counter += 1
                print(f"\n{'='*60}")
                print(f"Config {config_counter}/{total_configs}: "
                      f"Fold {fold_id} | {bert_name} | p={p}")
                print(f"{'='*60}")

                config_start = time.time()

                # Reset GPU memory tracking
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)

                # Apply coverage-aware fusion (Eq. 4)
                drug_fused = create_fused_drug_embeddings(
                    drug_s3_scaled[fold_id],
                    bert_scaled[bert_name],
                    p
                )

                has_bert = ~np.all(bert_scaled[bert_name] == 0, axis=1)
                print(f"  Fusion: {p*100:.0f}% GNN + {(1-p)*100:.0f}% BERT, "
                      f"Coverage: {has_bert.sum()}/{len(has_bert)}")

                # Initialize GNN with fused embeddings via Embedding encoder weights
                model = HeteroGNN(
                    num_drugs=num_drugs,
                    num_diseases=num_diseases,
                    num_go_terms=num_go_terms,
                    hidden_channels=HIDDEN_CHANNELS,
                    num_layers=NUM_LAYERS,
                    dropout=DROPOUT,
                    use_learnable_embeddings=True
                ).to(device)

                with torch.no_grad():
                    model.drug_encoder.weight.data = torch.from_numpy(
                        drug_fused).float().to(device)
                    model.disease_encoder.weight.data = torch.from_numpy(
                        disease_s3_scaled[fold_id]).float().to(device)
                    model.go_encoder.weight.data = torch.from_numpy(
                        go_s3_scaled[fold_id]).float().to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                hetero_graph_device = hetero_graph.to(device)

                # Training loop with combined objective (Eq. 3)
                emb_times = []
                train_start = time.time()

                for epoch in range(NUM_EPOCHS):
                    model.train()
                    optimizer.zero_grad()

                    embeddings, emb_time = generate_embeddings(
                        model, hetero_graph_device, device=device, training=True
                    )
                    emb_times.append(emb_time)

                    drug_emb = embeddings['drug']
                    disease_emb = embeddings['disease']

                    bce_loss = compute_bce_loss(
                        drug_emb, disease_emb, positive_pairs,
                        num_neg_samples=len(positive_pairs)
                    )
                    contrastive_loss = compute_pucl_contrastive_loss(
                        drug_emb, disease_emb, positive_pairs,
                        temperature=CONTRASTIVE_TEMP
                    )
                    total_loss = bce_loss + CONTRASTIVE_WEIGHT * contrastive_loss

                    total_loss.backward()
                    optimizer.step()

                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        print(f"  Epoch {epoch+1:02d}: Loss = {total_loss.item():.4f} "
                              f"(BCE = {bce_loss.item():.4f}, "
                              f"Contrastive = {contrastive_loss.item():.4f})")

                train_end = time.time()

                # Extract final embeddings for downstream evaluation
                model.eval()
                with torch.no_grad():
                    final_embeddings, _ = generate_embeddings(
                        model, hetero_graph_device, device=device, training=False
                    )
                    drug_emb_final = final_embeddings['drug']
                    disease_emb_final = final_embeddings['disease']

                # Downstream evaluation with PU-bagging
                downstream_start = time.time()

                X_train, y_train = build_training_set(
                    train_drug_disease_matrix=train_drug_disease_matrix,
                    drug_embeddings=drug_emb_final,
                    disease_embeddings=disease_emb_final
                )
                X_test, y_test = build_test_set(
                    test_edges=test_edges,
                    drug_embeddings=drug_emb_final,
                    disease_embeddings=disease_emb_final,
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
                config_end = time.time()

                # Record metadata
                metrics['fold_id'] = fold_id
                metrics['bert_model'] = bert_name
                metrics['p'] = p
                metrics['train_sec'] = train_end - train_start
                metrics['downstream_sec'] = downstream_end - downstream_start
                metrics['total_sec'] = config_end - config_start

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    max_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
                    metrics['max_gpu_mem_mb'] = max_mem

                all_results.append(metrics)

                print(f"\n  Results: AUPRC = {metrics['AUPRC']:.4f}, "
                      f"AUROC = {metrics['AUROC']:.4f}, "
                      f"F1_PU = {metrics['F1_PU']:.4f}, "
                      f"Time = {metrics['total_sec']:.1f}s")

                # Free GPU memory between configurations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Summary: Setting 4, Phase {phase}")
    print(f"{'='*60}")

    for bert_name in BERT_EMBEDDING_FILES:
        print(f"\n  {bert_name}:")
        for p in P_VALUES:
            subset = [m for m in all_results
                       if m['bert_model'] == bert_name and m['p'] == p]
            if subset:
                auprc_vals = [m['AUPRC'] for m in subset]
                auroc_vals = [m['AUROC'] for m in subset]
                f1pu_vals = [m['F1_PU'] for m in subset]
                print(f"    p={p}: AUPRC = {np.mean(auprc_vals):.3f} "
                      f"+/- {np.std(auprc_vals):.3f}, "
                      f"AUROC = {np.mean(auroc_vals):.3f} "
                      f"+/- {np.std(auroc_vals):.3f}, "
                      f"F1_PU = {np.mean(f1pu_vals):.3f} "
                      f"+/- {np.std(f1pu_vals):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Setting 4 (BERT fusion) for drug-disease association prediction."
    )
    parser.add_argument(
        "--phase", type=int, default=2, choices=[1, 2],
        help="Dataset phase (1 = 2019 data, 2 = 2025 data). Default: 2."
    )
    args = parser.parse_args()

    main(phase=args.phase)
