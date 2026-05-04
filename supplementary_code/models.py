"""
models.py — GNN Model Architecture and Loss Functions

This module defines the heterogeneous graph neural network (HeteroGNN)
used for drug-disease association prediction, along with the loss
functions for supervised training (BCE) and positive-unlabeled
contrastive learning (puCL).

Reference:
    T. Sinsrangboon and T. Panitanarak, "A Linearly Scalable GNN
    Framework on Drug-Gene Ontology-Disease Tripartite Graphs for
    Drug-Disease Association Prediction With Positive-Unlabeled
    Contrastive Learning," IEEE Access, 2026.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, LayerNorm, Embedding
from torch_geometric.nn import GraphConv, HeteroConv


# ============================================================================
# Model Architecture
# ============================================================================

class HeteroGNN(torch.nn.Module):
    """
    Heterogeneous Graph Neural Network for the drug-GO-disease tripartite graph.

    The model supports two input modes:
      - Learnable embeddings (use_learnable_embeddings=True):
        Node indices are mapped to dense vectors via nn.Embedding layers.
        Used in Setting 4 where pre-trained embeddings initialize the weights.
      - One-hot features (use_learnable_embeddings=False):
        One-hot identity vectors are projected via nn.Linear layers.
        Used in Settings 1-3.

    Architecture:
      1. Node-type-specific encoders project inputs to hidden_channels dimensions.
      2. Multiple HeteroConv layers perform message passing across all edge types.
         Each layer applies GraphConv with sum aggregation, followed by LayerNorm,
         dropout, and a residual (skip) connection.
      3. Final embeddings are computed as the mean across all layer outputs
         (including the initial encoding), providing multi-scale representations.

    Args:
        num_drugs (int): Number of drug nodes.
        num_diseases (int): Number of disease nodes.
        num_go_terms (int): Number of Gene Ontology term nodes.
        hidden_channels (int): Dimensionality of node embeddings (default: 128).
        num_layers (int): Number of message-passing layers (default: 2).
        dropout (float): Dropout rate applied after each layer (default: 0.2).
        use_learnable_embeddings (bool): If True, use nn.Embedding encoders;
            otherwise, use nn.Linear encoders for one-hot inputs (default: True).
    """

    def __init__(
        self,
        num_drugs,
        num_diseases,
        num_go_terms,
        hidden_channels=128,
        num_layers=2,
        dropout=0.2,
        use_learnable_embeddings=True
    ):
        super().__init__()
        self._hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_learnable_embeddings = use_learnable_embeddings

        # Node-type-specific encoders
        if use_learnable_embeddings:
            self.drug_encoder = Embedding(num_drugs, hidden_channels)
            self.disease_encoder = Embedding(num_diseases, hidden_channels)
            self.go_encoder = Embedding(num_go_terms, hidden_channels)
        else:
            self.drug_encoder = Linear(num_drugs, hidden_channels)
            self.disease_encoder = Linear(num_diseases, hidden_channels)
            self.go_encoder = Linear(num_go_terms, hidden_channels)

        # Heterogeneous message-passing layers
        self.convs = ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('drug', 'connects', 'go'): GraphConv(hidden_channels, hidden_channels),
                ('go', 'rev_connects', 'drug'): GraphConv(hidden_channels, hidden_channels),
                ('disease', 'connects', 'go'): GraphConv(hidden_channels, hidden_channels),
                ('go', 'rev_connects', 'disease'): GraphConv(hidden_channels, hidden_channels),
                ('drug', 'connects', 'disease'): GraphConv(hidden_channels, hidden_channels),
                ('disease', 'rev_connects', 'drug'): GraphConv(hidden_channels, hidden_channels)
            }, aggr='sum')
            self.convs.append(conv)

        # Layer normalization for each message-passing layer
        self.layer_norms = ModuleList([
            LayerNorm(hidden_channels) for _ in range(num_layers)
        ])

        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the heterogeneous GNN.

        Args:
            x_dict (dict): Node features keyed by node type.
                For Embedding mode: integer indices per node type.
                For Linear mode: one-hot float vectors per node type.
            edge_index_dict (dict): Edge indices keyed by edge type tuple.

        Returns:
            dict: Final node embeddings keyed by node type ('drug', 'disease', 'go').
        """
        # Initial encoding
        if self.use_learnable_embeddings:
            hidden_dict = {
                'drug': self.drug_encoder(x_dict['drug'].squeeze()),
                'disease': self.disease_encoder(x_dict['disease'].squeeze()),
                'go': self.go_encoder(x_dict['go'].squeeze())
            }
        else:
            hidden_dict = {
                'drug': self.drug_encoder(x_dict['drug'].float()),
                'disease': self.disease_encoder(x_dict['disease'].float()),
                'go': self.go_encoder(x_dict['go'].float())
            }

        # Collect embeddings from all layers for mean pooling
        all_embeddings = {
            node_type: [hidden_dict[node_type]]
            for node_type in hidden_dict
        }

        # Message passing with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            hidden_dict_new = conv(hidden_dict, edge_index_dict)

            for node_type in hidden_dict:
                emb_new = hidden_dict_new.get(node_type)
                if emb_new is None:
                    # Node type not updated by this layer; retain previous embedding
                    emb_new = hidden_dict[node_type]
                else:
                    emb_new = norm(emb_new)
                    emb_new = F.dropout(emb_new, p=self.dropout, training=self.training)
                    emb_new = emb_new + hidden_dict[node_type]  # Residual connection

                hidden_dict[node_type] = emb_new
                all_embeddings[node_type].append(emb_new)

        # Mean pooling across all layers (initial + message-passing layers)
        final_embeddings = {
            node_type: torch.mean(torch.stack(embs, dim=0), dim=0)
            for node_type, embs in all_embeddings.items()
        }

        return final_embeddings

    def reset_parameters(self):
        """Re-initialize all learnable parameters."""
        self.drug_encoder.reset_parameters()
        self.disease_encoder.reset_parameters()
        self.go_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.layer_norms:
            norm.reset_parameters()

    @property
    def hidden_channels(self):
        return self._hidden_channels


# ============================================================================
# Loss Functions
# ============================================================================

def compute_bce_loss(drug_emb, disease_emb, pos_pairs, num_neg_samples=10000, seed=42):
    """
    Compute binary cross-entropy loss for drug-disease link prediction.

    Positive pairs are scored using dot-product similarity followed by sigmoid.
    An equal number of unlabeled (assumed negative) pairs are sampled randomly
    to balance the gradient contributions.

    Args:
        drug_emb (Tensor): Drug embeddings, shape (num_drugs, dim).
        disease_emb (Tensor): Disease embeddings, shape (num_diseases, dim).
        pos_pairs (np.ndarray): Known positive drug-disease pairs, shape (N, 2).
        num_neg_samples (int): Number of negative samples to draw (default: 10000).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        Tensor: Scalar BCE loss value.
    """
    rng = np.random.default_rng(seed)

    # Score positive pairs
    pos_i = torch.tensor(pos_pairs[:, 0], device=drug_emb.device)
    pos_j = torch.tensor(pos_pairs[:, 1], device=drug_emb.device)
    pos_scores = torch.sigmoid((drug_emb[pos_i] * disease_emb[pos_j]).sum(dim=1))
    pos_labels = torch.ones_like(pos_scores)

    # Sample and score negative (unlabeled) pairs
    num_drugs, num_diseases = drug_emb.shape[0], disease_emb.shape[0]
    pos_set = set(map(tuple, pos_pairs))
    neg_samples = set()

    while len(neg_samples) < num_neg_samples:
        i = rng.integers(0, num_drugs)
        j = rng.integers(0, num_diseases)
        if (i, j) not in pos_set:
            neg_samples.add((i, j))

    neg_samples = np.array(list(neg_samples))
    neg_i = torch.tensor(neg_samples[:, 0], device=drug_emb.device)
    neg_j = torch.tensor(neg_samples[:, 1], device=drug_emb.device)
    neg_scores = torch.sigmoid((drug_emb[neg_i] * disease_emb[neg_j]).sum(dim=1))
    neg_labels = torch.zeros_like(neg_scores)

    # Combined BCE loss
    all_scores = torch.cat([pos_scores, neg_scores])
    all_labels = torch.cat([pos_labels, neg_labels])
    bce_loss = F.binary_cross_entropy(all_scores, all_labels)

    return bce_loss


def compute_pucl_contrastive_loss(drug_embeddings, disease_embeddings,
                                   positive_pairs, temperature=0.1):
    """
    Compute adapted Positive-Unlabeled Contrastive Loss (puCL) for
    heterogeneous drug-disease link prediction.

    For each drug with known positive associations, the loss encourages
    high similarity between the drug embedding and its paired disease
    embeddings relative to all other diseases in the batch.

    Inspired by the puCL framework of Acharya et al. (2025), adapted
    for cross-domain contrastive learning without augmentation.

    Args:
        drug_embeddings (Tensor): Drug embeddings, shape (num_drugs, dim).
        disease_embeddings (Tensor): Disease embeddings, shape (num_diseases, dim).
        positive_pairs (np.ndarray): Known positive pairs, shape (N, 2).
        temperature (float): Temperature scaling parameter (default: 0.1).

    Returns:
        Tensor: Scalar contrastive loss value.
    """
    drug_emb = F.normalize(drug_embeddings, dim=-1)
    disease_emb = F.normalize(disease_embeddings, dim=-1)

    # Map each drug to its list of known positive diseases
    drug_to_pos = {}
    for d_idx, dis_idx in positive_pairs:
        drug_to_pos.setdefault(d_idx, []).append(dis_idx)

    if len(drug_to_pos) == 0:
        return torch.tensor(0.0, device=drug_emb.device, requires_grad=True)

    total_loss = 0.0

    for drug_idx, pos_disease_list in drug_to_pos.items():
        anchor = drug_emb[drug_idx]

        # Cosine similarity with all diseases, scaled by temperature
        sims = torch.matmul(anchor, disease_emb.T) / temperature
        log_denominator = torch.logsumexp(sims, dim=-1)

        # Mean similarity with known positive diseases
        pos_idx = torch.tensor(pos_disease_list, device=anchor.device)
        pos_mean_sim = sims[pos_idx].mean()

        # InfoNCE-style loss for this anchor
        loss_i = -(pos_mean_sim - log_denominator)
        total_loss += loss_i

    return total_loss / len(drug_to_pos)
