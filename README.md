# GNN-puCL-DDA

[![Status](https://img.shields.io/badge/Status-Accepted-green)](https://ieeeaccess.ieee.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for the paper **"A Linearly Scalable GNN Framework on Drug–Gene Ontology–Disease Tripartite Graphs for Drug–Disease Association Prediction With Positive-Unlabeled Contrastive Learning"**.

Accepted at **IEEE Access**, 2026.

## Overview

In drug-disease association prediction, graph neural networks (GNNs) offer a powerful means to maintain linear complexity without compromising high predictive accuracy. This repository provides the implementation of a scalable GNN framework that introduces a **positive-unlabeled contrastive learning (puCL)** objective and a **coverage-aware fusion strategy**.

The model addresses the inherent bias caused by assuming unlabeled pairs are true negatives, and successfully integrates semantic knowledge from biomedical language models (PubMedBERT and BioBERT).

## Key Contributions

- **Linearly Scalable Framework:** GNN with linear O(|E|) complexity, solving the high computational costs of meta-path enumeration methods.
- **puCL Objective:** Clusters known associations to refine the latent space geometry and address the inherent bias of incomplete biological data.
- **Semantic Integration:** Coverage-aware fusion strategy leveraging biomedical language models, remaining robust even when textual descriptions are missing for 42% of compounds.
- **Temporal Robustness:** Validated on datasets spanning six years of biomedical progress, maintaining 99.0% AUPRC despite a 7.4-fold growth in network edges.

## Repository Structure

```
GNN-puCL-DDA/
├── predictions/
│   └── unlabeled_pairs_prob_ge_0.6.csv   # Scored unlabeled pairs (prob ≥ 0.6, Setting 3, Phase 2)
├── supplementary_code/
│   ├── models.py                          # HeteroGNN architecture and loss functions
│   ├── utils.py                           # Graph construction, embedding generation, downstream processing
│   ├── run_setting3.py                    # Setting 3: supervised training with puCL objective
│   ├── run_setting4.py                    # Setting 4: puCL + BERT semantic fusion
│   ├── feature_importance.py             # XGBoost feature importance analysis (drug vs disease embeddings)
│   └── predict_unlabeled.py              # Full retrain + score all unlabeled pairs → CSV
├── .gitignore
├── LICENSE
└── README.md
```

## Experimental Settings

| Setting | Description |
|---------|-------------|
| 1 | Structure-only propagation (untrained GNN) |
| 2 | Supervised training with BCE loss |
| 3 | Setting 2 + puCL contrastive objective |
| 4 | Setting 3 + BERT semantic fusion |

## Predicted Associations

High-confidence predicted drug–disease associations from Setting 3 (trained on all Phase 2 associations) are available in [predictions/unlabeled_pairs_prob_ge_0.6.csv](predictions/unlabeled_pairs_prob_ge_0.6.csv). The file contains all unlabeled pairs scoring ≥ 0.6, with columns:

```
drugbank_id, disease_umls_cui, probability
```

Sorted by probability in descending order.

## Setup and Usage

See `supplementary_code/README.md` for environment setup and instructions to run each experimental setting.

## Authors

- **Thanapat Sinsrangboon** — Department of Mathematics and Computer Science, Faculty of Science, Chulalongkorn University, Thailand
- **Thap Panitanarak** — Department of Mathematics and Computer Science, Faculty of Science, Chulalongkorn University, Thailand

## Acknowledgments

This work was supported by the **Development and Promotion of Science and Technology Talents Project (DPST)**, Thailand. The authors utilized Claude (Anthropic) for manuscript drafting and structuring, and Gemini (Google) for content review and refinement. All technical design, experimental implementation, and scientific conclusions were developed independently by the authors.

## Citation

If you find this work useful, please cite:

```bibtex
@article{sinsrangboon2026gnn,
  title={A Linearly Scalable GNN Framework on Drug--Gene Ontology--Disease Tripartite Graphs
         for Drug--Disease Association Prediction With Positive-Unlabeled Contrastive Learning},
  author={Sinsrangboon, Thanapat and Panitanarak, Thap},
  journal={IEEE Access},
  year={2026},
  note={Accepted}
}
```
