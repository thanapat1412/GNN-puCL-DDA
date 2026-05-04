# Supplementary Code

**A Linearly Scalable GNN Framework on Drug–Gene Ontology–Disease Tripartite Graphs for Drug–Disease Association Prediction With Positive-Unlabeled Contrastive Learning**

T. Sinsrangboon and T. Panitanarak, IEEE Access, 2026.

## Overview

This repository contains the source code for the experimental settings presented in the paper. The framework predicts drug–disease associations using a heterogeneous GNN on a drug–GO–disease tripartite graph, evaluated through four progressive settings:

| Setting | Description | Script |
|---------|-------------|--------|
| 1 | Structure-only propagation (untrained GNN) | `run_setting3.py --setting 1` |
| 2 | Supervised training with BCE loss (Eq. 1) | `run_setting3.py --setting 2` |
| 3 | Setting 2 + puCL contrastive learning (Eq. 3) | `run_setting3.py --setting 3` |
| 4 | Setting 3 + coverage-aware BERT fusion (Eq. 4) | `run_setting4.py` |

## File Structure

```
supplementary_code/
├── README.md              # This file
├── models.py              # HeteroGNN architecture and loss functions
├── utils.py               # Graph construction, embeddings, PU-bagging classifier
├── run_setting3.py        # Main script for Settings 1-3
└── run_setting4.py        # Main script for Setting 4 (BERT fusion)
```

## Requirements

- Python 3.10
- PyTorch 2.8.0 (CUDA 12.6)
- PyTorch Geometric (with torch-scatter, torch-sparse, pyg-lib)
- XGBoost
- scikit-learn
- NumPy
- Pandas

The original experiments were conducted on Google Colab with NVIDIA Tesla T4 GPUs.

## Data Preparation

### Tripartite Network Data

The adjacency matrices and node lists are derived from four public biomedical databases. Two temporal snapshots are used: Phase 1 (2019 versions) and Phase 2 (2025 versions).

**Phase 2 database versions:** DrugBank v5.1.13, DisGeNET v25.3, GOA release 228, CTD (July 2025).

Place the following files in `./data/`:

```
data/
├── drug_go_matrix_Phase{1,2}.txt         # Drug-GO adjacency (tab-delimited, int8)
├── disease_go_matrix_Phase{1,2}.txt      # Disease-GO adjacency (tab-delimited, int8)
├── drug_disease_matrix_Phase{1,2}.txt    # Drug-disease adjacency (tab-delimited, int8)
├── drug_list_Phase{1,2}.csv              # Drug node list (column: drugBankId)
├── disease_list_Phase{1,2}.csv           # Disease node list (column: diseaseUMLSCUI)
├── go_union_list_Phase{1,2}.csv          # GO term list (column: goId)
└── folds_5folds_Phase{1,2}.pkl           # Pre-computed 5-fold CV splits
```

The network is constructed by extracting overlapping entities across the four databases to form the tripartite structure (drug–GO, disease–GO, drug–disease).

### BERT Embeddings (Setting 4 only)

Drug mechanism descriptions were encoded using two biomedical language models:

- **PubMedBERT**: `NeuML/pubmedbert-base-embeddings`
- **BioBERT**: `dmis-lab/biobert-base-cased-v1.2`

The encoding pipeline produces 768-dimensional vectors, which are then reduced to 128 dimensions via PCA (fitted only on drugs with descriptions; drugs without descriptions retain zero vectors). Setting 4 further reduces these to 64 dimensions to match the GNN embedding space.

Place the BERT embedding files in `./data/`:

```
data/
├── drug_pubmedbert_embeddings_128d.npy   # PubMedBERT (num_drugs, 128)
└── drug_biobert_embeddings_128d.npy      # BioBERT (num_drugs, 128)
```

## Usage

### Settings 1-3

```bash
# Setting 1: Structure-only (no training)
python run_setting3.py --setting 1 --phase 2

# Setting 2: Supervised BCE training
python run_setting3.py --setting 2 --phase 2

# Setting 3: BCE + puCL contrastive learning
python run_setting3.py --setting 3 --phase 2
```

Setting 3 saves embeddings to `./outputs/embeddings/` which are required by Setting 4.

### Setting 4

```bash
# Run after Setting 3 completes
python run_setting4.py --phase 2
```

This evaluates all combinations of BERT models (PubMedBERT, BioBERT) and fusion weights (p = 0.3, 0.5, 0.7).
