# GNN-puCL-DDA

[![Conference](https://img.shields.io/badge/Status-Submitted-blue)](https://ieeeaccess.ieee.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for the paper **"A Linearly Scalable GNN Framework on Drug-Gene Ontology-Disease Tripartite Graphs for Drug-Disease Association Prediction With Positive-Unlabeled Contrastive Learning"**[cite: 1].

Currently under review at **IEEE Access**[cite: 1].

## 📖 Overview

In drug-disease association prediction, graph neural networks (GNNs) offer a powerful means to maintain linear complexity without compromising high predictive accuracy[cite: 1]. This repository provides the implementation of a scalable GNN framework that introduces a **positive-unlabeled contrastive learning (puCL)** objective and a **coverage-aware fusion strategy**[cite: 1]. 

The model addresses the inherent bias caused by assuming unlabeled pairs are true negatives, and successfully integrates semantic knowledge from biomedical language models (PubMedBERT and BioBERT)[cite: 1].

## ✨ Key Contributions

*   **Linearly Scalable Framework:** Employs a GNN framework with linear $O(|E|)$ complexity to effectively solve the high computational costs[cite: 1].
*   **puCL Objective:** Clusters known associations to refine the latent space geometry and address the inherent bias of incomplete biological data[cite: 1].
*   **Semantic Integration:** Leverages biomedical language models through a coverage-aware fusion strategy, remaining robust even when textual descriptions are missing for 42% of the compounds[cite: 1].
*   **Temporal Robustness:** Validated on datasets representing six years of biomedical progress, maintaining a 99.0% AUPRC despite a 7.4-fold growth in network edges[cite: 1].

## 📂 Repository Structure

The source code and execution scripts are located in the `supplementary_code/` directory:

*   `models.py`: Contains the HeteroGNN architecture and loss functions.
*   `utils.py`: Includes graph construction, embedding generation, and downstream processing utilities.
*   `run_setting3.py`: Script to execute Setting 3 (Supervised training with puCL objective).
*   `run_setting4.py`: Script to execute Setting 4 (puCL framework combined with BERT semantic fusion).

*(Detailed instructions on how to set up the environment and run these scripts can be found in `supplementary_code/README.md`)*

## 👥 Authors

*   **Thanapat Sinsrangboon** - Department of Mathematics and Computer Science, Faculty of Science, Chulalongkorn University, Thailand[cite: 1].
*   **Thap Panitanarak** - Department of Mathematics and Computer Science, Faculty of Science, Chulalongkorn University, Thailand[cite: 1].

## 🙏 Acknowledgments

This work was supported by the Development and Promotion of Science and Technology Talents Project (DPST), Thailand[cite: 1].

## 📝 Citation

If you find this work or code useful, please cite our paper:
*To Submitting*
```bibtex
@article{sinsrangboon2026gnn,
  title={A Linearly Scalable GNN Framework on Drug-Gene Ontology-Disease Tripartite Graphs for Drug-Disease Association Prediction With Positive-Unlabeled Contrastive Learning},
  author={Sinsrangboon, Thanapat and Panitanarak, Thap},
  journal={IEEE Access},
  year={2026},
  note={Submitted}
}
