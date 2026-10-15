"""
Predict Unlabeled Drug-Disease Associations — Setting 3 (Full Retrain)
=======================================================================
Retrains the GNN and PU-bagging classifier on ALL known associations
(no edge masking), scores every unlabeled drug-disease pair, and exports
pairs above a probability threshold to CSV.

This follows the prediction protocol used in Section V-E of the paper:
CV evaluation (Table 3) measures model quality with edge masking;
this script deploys the model on full data to generate candidate predictions.

Setting : Setting 3 (puCL GNN)
Phase   : Phase 2 (2025 databases)
Output  : CSV with columns drugbank_id, disease_umls_cui, probability
          filtered to pairs >= --threshold (default 0.6)

Usage
-----
python predict_unlabeled.py \
    --drug_go_matrix      drug_go_matrix_Phase2.txt \
    --disease_go_matrix   disease_go_matrix_Phase2.txt \
    --drug_disease_matrix drug_disease_matrix_Phase2.txt \
    --drug_list           drug_list_Phase2.csv \
    --disease_list        disease_list_Phase2.csv \
    --go_list             go_union_list_Phase2.csv \
    --output_path         unlabeled_pairs_prob_ge_0.6.csv \
    --threshold           0.6 \
    --embeddings_cache    embeddings_cache/ \
    --batch_cache         batch_cache/
"""

import argparse
import gc
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb
from torch.nn import Embedding, LayerNorm, Linear, ModuleList
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, HeteroConv


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_heterogeneous_graph(
    drug_go_matrix: np.ndarray,
    disease_go_matrix: np.ndarray,
    drug_disease_matrix: np.ndarray,
) -> HeteroData:
    """Build HeteroData graph from adjacency matrices."""
    hetero_graph = HeteroData()

    num_drugs = drug_go_matrix.shape[0]
    num_diseases = disease_go_matrix.shape[0]
    num_go_terms = drug_go_matrix.shape[1]

    hetero_graph["drug"].x = torch.eye(num_drugs, dtype=torch.float)
    hetero_graph["disease"].x = torch.eye(num_diseases, dtype=torch.float)
    hetero_graph["go"].x = torch.eye(num_go_terms, dtype=torch.float)

    drug_idx, go_idx = np.where(drug_go_matrix)
    hetero_graph["drug", "connects", "go"].edge_index = torch.tensor(
        [drug_idx, go_idx], dtype=torch.long
    )
    hetero_graph["go", "rev_connects", "drug"].edge_index = torch.tensor(
        [go_idx, drug_idx], dtype=torch.long
    )

    dis_idx, go_idx = np.where(disease_go_matrix)
    hetero_graph["disease", "connects", "go"].edge_index = torch.tensor(
        [dis_idx, go_idx], dtype=torch.long
    )
    hetero_graph["go", "rev_connects", "disease"].edge_index = torch.tensor(
        [go_idx, dis_idx], dtype=torch.long
    )

    drug_idx, dis_idx = np.where(drug_disease_matrix)
    hetero_graph["drug", "connects", "disease"].edge_index = torch.tensor(
        [drug_idx, dis_idx], dtype=torch.long
    )
    hetero_graph["disease", "rev_connects", "drug"].edge_index = torch.tensor(
        [dis_idx, drug_idx], dtype=torch.long
    )

    return hetero_graph


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HeteroGNN(torch.nn.Module):
    """Heterogeneous GNN — identical to SETTING_3.py."""

    def __init__(
        self,
        num_drugs: int,
        num_diseases: int,
        num_go_terms: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.drug_encoder = Linear(num_drugs, hidden_channels)
        self.disease_encoder = Linear(num_diseases, hidden_channels)
        self.go_encoder = Linear(num_go_terms, hidden_channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("drug", "connects", "go"): GraphConv(hidden_channels, hidden_channels),
                    ("go", "rev_connects", "drug"): GraphConv(hidden_channels, hidden_channels),
                    ("disease", "connects", "go"): GraphConv(hidden_channels, hidden_channels),
                    ("go", "rev_connects", "disease"): GraphConv(hidden_channels, hidden_channels),
                    ("drug", "connects", "disease"): GraphConv(hidden_channels, hidden_channels),
                    ("disease", "rev_connects", "drug"): GraphConv(hidden_channels, hidden_channels),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.layer_norms = ModuleList(
            [LayerNorm(hidden_channels) for _ in range(num_layers)]
        )
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        hidden_dict = {
            "drug": self.drug_encoder(x_dict["drug"].float()),
            "disease": self.disease_encoder(x_dict["disease"].float()),
            "go": self.go_encoder(x_dict["go"].float()),
        }
        all_embeddings = {nt: [hidden_dict[nt]] for nt in hidden_dict}

        for conv, norm in zip(self.convs, self.layer_norms):
            hidden_dict_new = conv(hidden_dict, edge_index_dict)
            for node_type in hidden_dict:
                emb_new = hidden_dict_new.get(node_type, hidden_dict[node_type])
                emb_new = norm(emb_new)
                emb_new = F.dropout(emb_new, p=self.dropout, training=self.training)
                emb_new = emb_new + hidden_dict[node_type]
                hidden_dict[node_type] = emb_new
                all_embeddings[node_type].append(emb_new)

        return {
            nt: torch.mean(torch.stack(embs, dim=0), dim=0)
            for nt, embs in all_embeddings.items()
        }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_bce_loss(
    drug_emb: torch.Tensor,
    disease_emb: torch.Tensor,
    pos_pairs: np.ndarray,
    num_samples: int = 10000,
    seed: int = 42,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    pos_i = torch.tensor(pos_pairs[:, 0], device=drug_emb.device)
    pos_j = torch.tensor(pos_pairs[:, 1], device=drug_emb.device)
    pos_scores = torch.sigmoid((drug_emb[pos_i] * disease_emb[pos_j]).sum(dim=1))

    num_drugs, num_diseases = drug_emb.shape[0], disease_emb.shape[0]
    pos_set = set(map(tuple, pos_pairs))
    neg_samples: set = set()
    while len(neg_samples) < num_samples:
        i = rng.integers(0, num_drugs)
        j = rng.integers(0, num_diseases)
        if (i, j) not in pos_set:
            neg_samples.add((i, j))

    neg_arr = np.array(list(neg_samples))
    neg_scores = torch.sigmoid(
        (drug_emb[torch.tensor(neg_arr[:, 0], device=drug_emb.device)]
         * disease_emb[torch.tensor(neg_arr[:, 1], device=drug_emb.device)]).sum(dim=1)
    )
    all_scores = torch.cat([pos_scores, neg_scores])
    all_labels = torch.cat([
        torch.ones_like(pos_scores), torch.zeros_like(neg_scores)
    ])
    return F.binary_cross_entropy(all_scores, all_labels)


def compute_contrastive_loss(
    drug_emb: torch.Tensor,
    disease_emb: torch.Tensor,
    pos_pairs: np.ndarray,
    temperature: float = 0.1,
) -> torch.Tensor:
    drug_emb_n = F.normalize(drug_emb, dim=-1)
    disease_emb_n = F.normalize(disease_emb, dim=-1)

    drug_to_pos: dict = {}
    for d_idx, dis_idx in pos_pairs:
        drug_to_pos.setdefault(d_idx, []).append(dis_idx)

    if not drug_to_pos:
        return torch.tensor(0.0, device=drug_emb.device, requires_grad=True)

    total_loss = 0.0
    for drug_idx, pos_disease_list in drug_to_pos.items():
        anchor = drug_emb_n[drug_idx]
        sims = torch.matmul(anchor, disease_emb_n.T) / temperature
        logZ = torch.logsumexp(sims, dim=-1)
        pos_idx = torch.tensor(pos_disease_list, device=anchor.device)
        total_loss += -(sims[pos_idx].mean() - logZ)

    return total_loss / len(drug_to_pos)


# ---------------------------------------------------------------------------
# GNN training
# ---------------------------------------------------------------------------

def train_or_load_embeddings(
    drug_go_matrix: np.ndarray,
    disease_go_matrix: np.ndarray,
    drug_disease_matrix: np.ndarray,
    cache_dir: str,
    device: torch.device,
    epochs: int = 50,
    lambda_cl: float = 1.0,
) -> tuple:
    """Train GNN on full data or load cached embeddings."""
    emb_drug_path = os.path.join(cache_dir, "Drug_Embeddings_Setting3_FullTrain.npy")
    emb_disease_path = os.path.join(cache_dir, "Disease_Embeddings_Setting3_FullTrain.npy")

    if os.path.exists(emb_drug_path) and os.path.exists(emb_disease_path):
        print("Embeddings found in cache — loading (skip GNN training)")
        drug_emb_np = np.load(emb_drug_path)
        disease_emb_np = np.load(emb_disease_path)
        print(f"  Drug: {drug_emb_np.shape}, Disease: {disease_emb_np.shape}")
        return drug_emb_np, disease_emb_np

    print(f"Training GNN on full data ({drug_disease_matrix.sum():,} positive edges)...")
    hetero_graph = create_heterogeneous_graph(
        drug_go_matrix, disease_go_matrix, drug_disease_matrix
    )

    model = HeteroGNN(
        num_drugs=drug_go_matrix.shape[0],
        num_diseases=disease_go_matrix.shape[0],
        num_go_terms=drug_go_matrix.shape[1],
        hidden_channels=64,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_pairs = np.argwhere(drug_disease_matrix == 1)
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        g = hetero_graph.to(device)
        embs = model(g.x_dict, g.edge_index_dict)
        bce = compute_bce_loss(embs["drug"], embs["disease"], pos_pairs,
                               num_samples=len(pos_pairs))
        cl = compute_contrastive_loss(embs["drug"], embs["disease"], pos_pairs)
        loss = bce + lambda_cl * cl
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs}: loss={loss.item():.4f} "
                  f"(BCE={bce.item():.4f}, CL={cl.item():.4f})")

    print(f"  Training done in {time.time()-t0:.1f}s")

    drug_emb_np = embs["drug"].detach().cpu().numpy()
    disease_emb_np = embs["disease"].detach().cpu().numpy()
    os.makedirs(cache_dir, exist_ok=True)
    np.save(emb_drug_path, drug_emb_np)
    np.save(emb_disease_path, disease_emb_np)
    print(f"  Embeddings saved to {cache_dir}")

    del model, hetero_graph, g, embs
    torch.cuda.empty_cache()
    gc.collect()

    return drug_emb_np, disease_emb_np


# ---------------------------------------------------------------------------
# PU-Bagging
# ---------------------------------------------------------------------------

XGB_PARAMS = {
    "learning_rate": 0.1,
    "n_estimators": 200,
    "max_depth": 9,
    "min_child_weight": 1,
    "eval_metric": "logloss",
    "device": "cuda",
}


def train_or_load_classifiers(
    drug_emb_np: np.ndarray,
    disease_emb_np: np.ndarray,
    drug_disease_matrix: np.ndarray,
    cache_dir: str,
    T: int = 50,
) -> list:
    """Train PU-bagging classifiers or load from cache."""
    clf_path = os.path.join(cache_dir, "pu_bagging_50_classifiers.pkl")

    if os.path.exists(clf_path):
        print("Classifiers found in cache — loading (skip PU-Bagging training)")
        with open(clf_path, "rb") as f:
            classifiers = pickle.load(f)
        print(f"  Loaded {len(classifiers)} classifiers")
        return classifiers

    print(f"Training PU-Bagging (T={T})...")
    set_seed(42)
    unl_d, unl_dis = np.where(drug_disease_matrix == 0)
    pos_d, pos_dis = np.where(drug_disease_matrix == 1)

    X_pos = np.concatenate(
        [drug_emb_np[pos_d], disease_emb_np[pos_dis]], axis=1
    ).astype(np.float32)

    classifiers = []
    t0 = time.time()

    for t in range(T):
        sample_idx = np.random.choice(len(unl_d), size=len(pos_d), replace=True)
        X_neg = np.concatenate(
            [drug_emb_np[unl_d[sample_idx]], disease_emb_np[unl_dis[sample_idx]]],
            axis=1,
        ).astype(np.float32)
        X_train = np.vstack([X_pos, X_neg])
        y_train = np.array([1] * len(X_pos) + [0] * len(X_neg))

        clf = xgb.XGBClassifier(random_state=t, **XGB_PARAMS)
        clf.fit(X_train, y_train)
        classifiers.append(clf)

        if (t + 1) % 10 == 0:
            print(f"  Trained {t+1}/{T} classifiers")
        del X_neg, X_train, y_train

    print(f"  PU-Bagging done in {time.time()-t0:.1f}s")
    os.makedirs(cache_dir, exist_ok=True)
    with open(clf_path, "wb") as f:
        pickle.dump(classifiers, f)
    print(f"  Classifiers saved to {clf_path}")
    del X_pos
    gc.collect()

    return classifiers


# ---------------------------------------------------------------------------
# Batched prediction
# ---------------------------------------------------------------------------

def predict_batched(
    drug_emb_np: np.ndarray,
    disease_emb_np: np.ndarray,
    drug_disease_matrix: np.ndarray,
    classifiers: list,
    batch_cache_dir: str,
    batch_size: int = 500_000,
) -> np.ndarray:
    """Score all unlabeled pairs in batches; resume from cache if interrupted."""
    unl_d, unl_dis = np.where(drug_disease_matrix == 0)
    num_unlabeled = len(unl_d)
    num_batches = (num_unlabeled + batch_size - 1) // batch_size
    os.makedirs(batch_cache_dir, exist_ok=True)

    existing = {
        b for b in range(num_batches)
        if os.path.exists(os.path.join(batch_cache_dir, f"batch_{b:04d}.npy"))
    }
    print(f"Unlabeled pairs: {num_unlabeled:,} | Batches: {num_batches} | "
          f"Completed: {len(existing)}")

    t0 = time.time()
    for b in range(num_batches):
        if b in existing:
            continue
        start, end = b * batch_size, min((b + 1) * batch_size, num_unlabeled)
        X_batch = np.concatenate(
            [drug_emb_np[unl_d[start:end]], disease_emb_np[unl_dis[start:end]]],
            axis=1,
        ).astype(np.float32)

        prob_batch = np.zeros(len(X_batch), dtype=np.float64)
        for clf in classifiers:
            prob_batch += clf.predict_proba(X_batch)[:, 1]
        prob_batch /= len(classifiers)

        np.save(os.path.join(batch_cache_dir, f"batch_{b:04d}.npy"), prob_batch)
        print(f"  Batch {b+1}/{num_batches}: {len(X_batch):,} pairs, "
              f"prob=[{prob_batch.min():.4f}, {prob_batch.max():.4f}], "
              f"{time.time()-t0:.1f}s elapsed")
        del X_batch, prob_batch
        gc.collect()

    print("Combining batch results...")
    all_probs = np.concatenate([
        np.load(os.path.join(batch_cache_dir, f"batch_{b:04d}.npy"))
        for b in range(num_batches)
    ])
    return all_probs, unl_d, unl_dis


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_results(
    all_probs: np.ndarray,
    unl_d: np.ndarray,
    unl_dis: np.ndarray,
    drug_list: pd.Series,
    disease_list: pd.Series,
    output_path: str,
    threshold: float,
) -> None:
    mask = all_probs >= threshold
    output = pd.DataFrame({
        "drugbank_id": drug_list.iloc[unl_d[mask]].values,
        "disease_umls_cui": disease_list.iloc[unl_dis[mask]].values,
        "probability": all_probs[mask],
    }).sort_values("probability", ascending=False).reset_index(drop=True)

    output.to_csv(output_path, index=False)
    print(f"\nExported {len(output):,} pairs (prob >= {threshold}) to {output_path}")

    print(f"\nProbability distribution (filtered set):")
    print(f"  Min:    {output['probability'].min():.4f}")
    print(f"  Median: {output['probability'].median():.4f}")
    print(f"  Max:    {output['probability'].max():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score all unlabeled drug-disease pairs using Setting 3 (full retrain)"
    )
    parser.add_argument("--drug_go_matrix",      required=True)
    parser.add_argument("--disease_go_matrix",   required=True)
    parser.add_argument("--drug_disease_matrix", required=True)
    parser.add_argument("--drug_list",           required=True,
                        help="CSV with column drugBankId")
    parser.add_argument("--disease_list",        required=True,
                        help="CSV with column diseaseUMLSCUI")
    parser.add_argument("--go_list",             required=True,
                        help="CSV with column goId")
    parser.add_argument("--output_path",         default="unlabeled_pairs_prob_ge_0.6.csv")
    parser.add_argument("--threshold",           type=float, default=0.6)
    parser.add_argument("--embeddings_cache",    default="embeddings_cache/",
                        help="Directory to cache GNN embeddings")
    parser.add_argument("--batch_cache",         default="batch_cache/",
                        help="Directory to cache prediction batches (crash-safe)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    drug_go_matrix      = np.loadtxt(args.drug_go_matrix,      delimiter="\t", dtype=np.int8)
    disease_go_matrix   = np.loadtxt(args.disease_go_matrix,   delimiter="\t", dtype=np.int8)
    drug_disease_matrix = np.loadtxt(args.drug_disease_matrix, delimiter="\t", dtype=np.int8)
    drug_list    = pd.read_csv(args.drug_list)["drugBankId"].reset_index(drop=True)
    disease_list = pd.read_csv(args.disease_list)["diseaseUMLSCUI"].reset_index(drop=True)

    print(f"Drugs: {len(drug_list)}, Diseases: {len(disease_list)}, "
          f"Positive pairs: {drug_disease_matrix.sum():,}")

    # Train or load GNN embeddings
    drug_emb_np, disease_emb_np = train_or_load_embeddings(
        drug_go_matrix, disease_go_matrix, drug_disease_matrix,
        cache_dir=args.embeddings_cache, device=device,
    )

    # Train or load PU-Bagging classifiers
    classifiers = train_or_load_classifiers(
        drug_emb_np, disease_emb_np, drug_disease_matrix,
        cache_dir=args.embeddings_cache,
    )

    # Score all unlabeled pairs
    all_probs, unl_d, unl_dis = predict_batched(
        drug_emb_np, disease_emb_np, drug_disease_matrix,
        classifiers, batch_cache_dir=args.batch_cache,
    )

    # Export filtered CSV
    export_results(
        all_probs, unl_d, unl_dis,
        drug_list, disease_list,
        output_path=args.output_path,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
