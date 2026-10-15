"""
Feature Importance Analysis for GNN-puCL-DDA
=============================================
Computes XGBoost feature importance (normalized gain via feature_importances_,
the default behavior in XGBoost 3.x) across the
PU-bagging ensemble to quantify the relative contribution of drug vs.
disease embeddings to the final prediction.

Setting : Setting 3 (puCL GNN)
Phase   : Phase 2 (2025 databases)
Method  : For each of 5 CV folds, retrain PU-bagging (T=50, XGBoost) on
          edge-masked embeddings, average feature_importances_ over the 50
          classifiers, then average over the 5 folds.

Usage
-----
python feature_importance.py \
    --embeddings_path /path/to/Embeddings/ \
    --matrix_path    drug_disease_matrix_Phase2.txt \
    --folds_path     folds_5folds_Phase2.pkl \
    --output_path    feature_importance_setting3_cv.npy
"""

import argparse
import os
import pickle

import numpy as np
import xgboost as xgb


XGB_PARAMS = {
    "learning_rate": 0.1,
    "n_estimators": 200,
    "max_depth": 9,
    "min_child_weight": 1,
    "eval_metric": "logloss",
    "device": "cuda",
}


def load_data(matrix_path: str, folds_path: str):
    drug_disease_matrix = np.loadtxt(matrix_path, delimiter="\t", dtype=np.int8)
    with open(folds_path, "rb") as f:
        folds = pickle.load(f)
    print(f"drug_disease_matrix: {drug_disease_matrix.shape}")
    print(f"Folds loaded: {len(folds)}")
    return drug_disease_matrix, folds


def compute_fold_importance(
    fold: dict,
    drug_disease_matrix: np.ndarray,
    embeddings_path: str,
    xgb_params: dict,
    T: int = 50,
) -> np.ndarray:
    fold_id = fold["fold_id"]
    np.random.seed(42 + fold_id)

    drug_emb = np.load(
        os.path.join(embeddings_path, f"Drug_Embeddings_Setting3_Fold{fold_id}.npy")
    )
    disease_emb = np.load(
        os.path.join(embeddings_path, f"Disease_Embeddings_Setting3_Fold{fold_id}.npy")
    )

    # Mask test edges to match CV evaluation protocol
    train_matrix = drug_disease_matrix.copy()
    for i, j in np.array(fold["test_edges"]):
        train_matrix[i, j] = 0

    # Build balanced training set
    pos_d, pos_dis = np.where(train_matrix == 1)
    X_pos = np.concatenate([drug_emb[pos_d], disease_emb[pos_dis]], axis=1)

    unl_d, unl_dis = np.where(train_matrix == 0)
    idx = np.random.choice(len(unl_d), size=len(pos_d), replace=False)
    X_unl = np.concatenate([drug_emb[unl_d[idx]], disease_emb[unl_dis[idx]]], axis=1)

    X_train = np.vstack([X_pos, X_unl]).astype(np.float32)
    y_train = np.array([1] * len(pos_d) + [0] * len(idx))
    print(f"  Fold {fold_id}: {len(pos_d)} pos + {len(idx)} unlabeled")

    # PU-Bagging: T iterations, average feature importance over classifiers
    pos_idx = np.where(y_train == 1)[0]
    unl_idx = np.where(y_train == 0)[0]
    fold_importances = []

    for t in range(T):
        s = np.random.choice(unl_idx, size=len(pos_idx), replace=True)
        idx_t = np.concatenate([pos_idx, s])
        clf = xgb.XGBClassifier(random_state=t, **xgb_params)
        clf.fit(X_train[idx_t], np.array([1] * len(pos_idx) + [0] * len(s)))
        fold_importances.append(clf.feature_importances_)
        if (t + 1) % 10 == 0:
            print(f"    T={t+1}/{T}")

    return np.mean(fold_importances, axis=0)


def print_summary(mean_imp: np.ndarray, top_k: int = 10) -> None:
    drug_pct = mean_imp[:64].sum() * 100
    dis_pct = mean_imp[64:].sum() * 100
    print(f"\nDrug embedding contribution:    {drug_pct:.1f}%")
    print(f"Disease embedding contribution: {dis_pct:.1f}%")

    top_idx = np.argsort(mean_imp)[::-1][:top_k]
    print(f"\nTop-{top_k} most important dimensions:")
    for rank, idx in enumerate(top_idx, 1):
        side = "Drug" if idx < 64 else "Disease"
        dim = idx if idx < 64 else idx - 64
        print(f"  {rank}. {side} dim {dim}: {mean_imp[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Feature importance analysis for GNN-puCL-DDA (Setting 3, Phase 2)"
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Directory containing Drug/Disease_Embeddings_Setting3_Fold*.npy",
    )
    parser.add_argument(
        "--matrix_path",
        type=str,
        default="drug_disease_matrix_Phase2.txt",
        help="Path to drug-disease adjacency matrix (tab-delimited)",
    )
    parser.add_argument(
        "--folds_path",
        type=str,
        default="folds_5folds_Phase2.pkl",
        help="Path to 5-fold split file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="feature_importance_setting3_cv.npy",
        help="Path to save mean importance array (.npy)",
    )
    args = parser.parse_args()

    drug_disease_matrix, folds = load_data(args.matrix_path, args.folds_path)

    all_fold_importances = []
    for fold in folds:
        print(f"\nFold {fold['fold_id']}...")
        fold_imp = compute_fold_importance(
            fold, drug_disease_matrix, args.embeddings_path, XGB_PARAMS
        )
        all_fold_importances.append(fold_imp)
        print(f"  Fold {fold['fold_id']} done")

    mean_imp = np.mean(all_fold_importances, axis=0)
    print_summary(mean_imp)

    np.save(args.output_path, mean_imp)
    print(f"\nSaved: {args.output_path}")


if __name__ == "__main__":
    main()
