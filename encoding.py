from models import GINEncoder
from Helper_functions import FocalLoss, balanced_class_weights
from Helper_functions import calculate_metrics
import copy
import torch
import torch.nn as nn
import numpy

def pre_train_GIN_encoder(
    data,
    train_perf_eval,
    val_perf_eval,
    num_classes=2,
    hidden_units=256,
    lr_encoder=5e-2,      # <- encoder LR
    lr_head=0.1,         # <- head LR
    wd_encoder=5e-4,      # <- encoder WD
    wd_head=0.0,          # <- head WD (often 0)
    epochs=100,
    embedding_dim=128,
    patience=None,
    min_delta=0.0,
    log_early_stop=False
):
    device = data.x.device

    # Construct GIN encoder model
    encoder = GINEncoder(
        num_node_features=data.num_node_features,
        hidden_units=hidden_units,
        embedding_dim=embedding_dim,
        num_layers=3,
        dropout=0.2
    ).to(device)

    # Linear head for warm-starting the encoder
    head = nn.Linear(embedding_dim, 2).to(device)

    # Optimiser with parameter groups: separate LR/WD for encoder vs head
    optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": lr_encoder, "weight_decay": wd_encoder},
            {"params": head.parameters(),    "lr": lr_head,    "weight_decay": wd_head},
        ]
    )

    alpha_weights = balanced_class_weights(data.y[train_perf_eval])
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.5, reduction='mean')

    encoder.train(); head.train()

    best_val_f1 = float('-inf')
    best_encoder_state = copy.deepcopy(encoder.state_dict())
    epochs_without_improvement = 0
    best_epoch = -1

    for epoch in range(epochs):
        optimizer.zero_grad()
        encoded_out = encoder(data)
        out = head(encoded_out)
        loss = criterion(out[train_perf_eval], data.y[train_perf_eval])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            encoder.eval(); head.eval()
            val_logits = head(encoder(data))
            val_pred = val_logits[val_perf_eval].argmax(dim=-1)
            val_metrics = calculate_metrics(
                data.y[val_perf_eval].cpu().numpy(),
                val_pred.cpu().numpy()
            )
            current_val_f1 = val_metrics['f1_illicit']
            improved = current_val_f1 > (best_val_f1 + min_delta)
            if improved:
                best_val_f1 = current_val_f1
                best_encoder_state = copy.deepcopy(encoder.state_dict())
                epochs_without_improvement = 0
                best_epoch = epoch + 1
            else:
                epochs_without_improvement += 1
            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Loss: {loss.item():.4f}, "
                f"Val illicit recall: {val_metrics['recall_illicit']:.4f}, "
                f"Val illicit F1: {val_metrics['f1_illicit']:.4f}"
            )
            encoder.train(); head.train()

        if patience and epochs_without_improvement >= patience:
            if log_early_stop:
                print(
                    f"Encoder early stopping at epoch {epoch + 1} "
                    f"(best F1: {best_val_f1:.4f} @ epoch {best_epoch})"
                )
            break

    encoder.load_state_dict(best_encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder, best_encoder_state

#%% XGB encoder

# XGBoost-based encoder that turns node features into embeddings using tree leaf patterns.
# Dependencies: xgboost, scikit-learn, torch
#   pip install xgboost scikit-learn

import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
import numpy as np

class XGBEncoder(nn.Module):
    """
    An encoder that:
      1) Trains an XGBoost classifier on node features (supervised).
      2) Transforms each node via the learned leaf indices (one-hot over trees).
      3) Projects to a fixed embedding_dim with TruncatedSVD.
      4) L2 normalises the result.

    API mirrors your GINEncoder where possible:
      - forward(data): returns embeddings for all nodes in `data.x`.
      - embed(data): alias for forward in eval mode.
      - fit(data, y): required once to train trees + projection.
    """

    def __init__(
        self,
        num_node_features: int,
        embedding_dim: int = 128,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        random_state: int = 42,
        tree_method: str = "hist",
        device: str = "Cuda" if torch.cuda.is_available() else "CPU"
    ):
        # Store config
        self.num_node_features = num_node_features
        self.embedding_dim = embedding_dim

        # Supervised learner
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            random_state=random_state,
            tree_method=tree_method,
            use_label_encoder=False,
            eval_metric="logloss",
            device=device
        )

        # Post-hoc feature mappers (fitted in .fit)
        self._onehot = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
        self._svd = TruncatedSVD(n_components=embedding_dim, random_state=random_state)

        # Internal flag
        self._is_fitted = False

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.is_cuda:
                x = x.cpu()
            x = x.numpy()
        return x

    def fit(self, data, y):
        """
        Train the XGB encoder.
          data: PyG Data with data.x (N x F). edge_index is unused.
          y:    labels as Tensor/ndarray shape (N,). Supervised target.

        After this call, forward()/embed() can be used to produce embeddings.
        """
        X = self._to_numpy(data.x)
        y = self._to_numpy(y).astype(np.int32).ravel()

        if X.shape[1] != self.num_node_features:
            raise ValueError(f"Expected {self.num_node_features} features, got {X.shape[1]}.")

        # 1) Train trees
        self.model.fit(X, y)

        # 2) Get leaf indices per tree (N x T)
        leaf_idx = self.model.apply(X)  # shape: (N, n_trees)
        if leaf_idx.ndim == 3:  # some xgboost versions return (K, N, T) for multi-class; collapse
            # Take first class’ leaves; with binary objective this should not occur, but be safe.
            leaf_idx = leaf_idx[0]

        # 3) One-hot encode leaf positions (sparse, large but efficient)
        leaf_ohe = self._onehot.fit_transform(leaf_idx)

        # 4) Project to embedding_dim with SVD (dense, low-dim)
        Z = self._svd.fit_transform(leaf_ohe)

        # 5) L2 normalise
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

        self._is_fitted = True
        return torch.tensor(Z, dtype=torch.float32, device=data.x.device)

    @torch.no_grad()
    def forward(self, data):
        """
        Produce embeddings for nodes in data.x using trained trees + projection.
        """
        if not self._is_fitted:
            raise RuntimeError("XGBEncoder must be fitted before calling forward(). Call .fit(data, y).")

        X = self._to_numpy(data.x)

        # Transform through leaf pipeline
        leaf_idx = self.model.apply(X)
        if leaf_idx.ndim == 3:
            leaf_idx = leaf_idx[0]
        leaf_ohe = self._onehot.transform(leaf_idx)
        Z = self._svd.transform(leaf_ohe)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

        return torch.tensor(Z, dtype=torch.float32, device=data.x.device)

    @torch.no_grad()
    def embed(self, data):
        # Mirror your GINEncoder API
        return self.forward(data)

    def to(self, device):
        """
        For API symmetry only. This encoder’s learned objects live on CPU.
        Returns self so you can write: encoder = XGBEncoder(...).to(data.x.device)
        """
        return self

    def train(self):
        # No-op for API symmetry
        return self

    def eval(self):
        # No-op for API symmetry
        return self
    
from torch_geometric.data import Data
# This function wraps the encoder with train-only fitting and returns a new Data object whose x are the embneddings, but which preserves edge_index, y, and device.
def make_xgbe_embeddings(
    encoder,                     # instance of XGBEncoder
    data: Data,                  # full PyG graph
    train_mask, val_mask, test_mask
) -> Data:
    """Fit encoder on train subset only, then embed all nodes and return a new Data with x=embeddings."""
    device = data.x.device
    # Build a *view* for training (no copy of edge_index; it is not used by XGB)
    x_train = data.x[train_mask]
    y_train = data.y[train_mask]
    data_train = Data(x=x_train, edge_index=None)  # edge_index unused by XGBEncoder

    # Fit trees + projection on train only
    encoder.fit(data_train, y_train)

    # Now produce embeddings for the *full* graph
    encoder.eval()
    with torch.no_grad():
        Z = encoder.embed(data)          # shape [N, embedding_dim], on device of data.x

    # Return a new Data object carrying the same graph but with embedded features
    emb_data = Data(
        x=Z.to(device),
        edge_index=data.edge_index,
        y=data.y,
        num_nodes=data.num_nodes
    ).to(device)

    return emb_data


import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_undirected
import networkx as nx


# -----------------------------
# Positional / bias encodings
# -----------------------------
def sinusoidal_time_encoding(delta_t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    delta_t: [E] (edge-relative time, e.g., seconds since t_ref; can be negative)
    Returns: [E, dim] sinusoidal features (no learnable params).
    """
    # Map time to [0, +] by absolute or scaled shift; here we keep sign and scale.
    # Normalise to seconds -> days if your timestamps are large; adapt as needed.
    # Small epsilon to avoid inf in later divisions:
    x = delta_t.view(-1, 1)  # [E,1]
    # Frequency bands:
    i = torch.arange(dim, device=delta_t.device, dtype=torch.float32).view(1, -1)
    denom = torch.pow(10000.0, (2 * (i // 2)) / dim)  # classic Transformer scaling
    z = x / denom
    enc = torch.empty(x.size(0), dim, device=delta_t.device, dtype=torch.float32)
    enc[:, 0::2] = torch.sin(z[:, 0::2])
    enc[:, 1::2] = torch.cos(z[:, 1::2])
    return enc


def spd_encoding(edge_index: torch.Tensor, num_nodes: int, K: int) -> torch.Tensor:
    """
    Shortest-path distance (SPD) clipped at K, for edges; returns an integer code per edge.
    For efficiency we compute distances via BFS on an nx graph once, then index per edge.
    """
    ei = edge_index
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = ei.t().tolist()
    G.add_edges_from(edges)

    # Precompute all-pairs shortest path up to K using multi-source BFS per node:
    # We'll store SPD(u,v) for edges only, which is cheap:
    spd = []
    for u, v in edges:
        try:
            d = nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            d = K + 1
        spd.append(min(d, K))
    return torch.tensor(spd, dtype=torch.long, device=edge_index.device)


# -----------------------------
# Temporal-union builder
# -----------------------------
def build_temporal_union(
    data_by_time: Dict[int, Data],
    t_ref: int,
    window: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge snapshots within [t_ref - window[0], t_ref + window[1]] into a union graph.

    Args:
      data_by_time: mapping {t: Data} where Data.edge_index are the edges at time t
      t_ref:       reference time step we are embedding for
      window:      (past, future) inclusive hops in time units of keys in data_by_time

    Returns:
      edge_index [2,E_union], edge_time [E_union] with relative time (t_edge - t_ref)
    """
    ts = []
    for t, d in data_by_time.items():
        if (t_ref - window[0]) <= t <= (t_ref + window[1]):
            ei = d.edge_index
            E = ei.size(1)
            ts.append((
                ei, torch.full((E,), fill_value=(t - t_ref), dtype=torch.long, device=ei.device)
            ))

    if not ts:
        # Fallback: empty graph
        return (torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long))

    eis, rels = zip(*ts)
    edge_index = torch.cat(eis, dim=1)
    edge_time = torch.cat(rels, dim=0)

    # (Optional) deduplicate multi-edges by concatenation; here we keep all edges (attention can handle duplicates).
    edge_index = to_undirected(edge_index)  # if your graph is directed, remove this.
    return edge_index, edge_time


# -----------------------------
# DGT-style Transformer encoder
# -----------------------------
class DGTEncoder(nn.Module):
    """
    Dynamic Graph Transformer encoder producing node embeddings.
    - Spatial bias: shortest-path distance (SPD) buckets up to K.
    - Temporal bias: sinusoidal encoding of relative edge time.
    - Backbone: PyG TransformerConv with additive attention bias.

    API:
      forward(data, t_ref=None, window=None, data_by_time=None, edge_time=None)
        -> embeddings [N, embedding_dim]
    """
    def __init__(
        self,
        num_node_features: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        spd_K: int = 4,
        time_bias_dim: int = 16,
        spd_bias_dim: int = 16
    ):
        super().__init__()
        self.emb_in = nn.Linear(num_node_features, hidden_dim)

        self.time_bias_dim = time_bias_dim
        self.spd_K = spd_K

        # Learnable embeddings for SPD buckets [0..K] plus "no-path" bucket (K+1)
        self.spd_bias = nn.Embedding(spd_K + 2, spd_bias_dim)

        # Project concatenated biases to per-head additive bias
        self.bias_mlp = nn.Sequential(
            nn.Linear(time_bias_dim + spd_bias_dim, heads),
        )

        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                dropout=dropout,
                beta=True,          # allow residual attention weighting
                edge_dim=heads,     # pass per-head additive bias via edge_attr
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def _edge_bias(
        self,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute per-edge, per-head additive attention bias.
        Returns edge_attr with shape [E, heads].
        """
        # Temporal encoding:
        time_enc = sinusoidal_time_encoding(edge_time.float(), self.time_bias_dim)  # [E, Tdim]

        # Spatial encoding via SPD buckets (computed on the union graph):
        spd = spd_encoding(edge_index, num_nodes=num_nodes, K=self.spd_K)          # [E]
        spd_enc = self.spd_bias(spd)                                                # [E, Sdim]

        bias = torch.cat([time_enc, spd_enc], dim=-1)                               # [E, T+S]
        edge_attr = self.bias_mlp(bias)                                             # [E, heads]
        return edge_attr

    def forward(
        self,
        data: Data,
        *,
        t_ref: Optional[int] = None,
        window: Optional[Tuple[int, int]] = None,
        data_by_time: Optional[Dict[int, Data]] = None,
        edge_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Two usage modes:
          (A) Snapshot/union mode (recommended for dynamic): supply t_ref, window, data_by_time.
              The method will build a temporal-union edge_index and relative edge_time internally.
          (B) Pre-built mode: supply data.edge_index and edge_time directly (for a given union graph).

        Returns:
          X_emb: [N, embedding_dim]
        """
        x = self.emb_in(data.x)  # [N, H]
        N = data.num_nodes

        if (t_ref is not None) and (window is not None) and (data_by_time is not None):
            edge_index, rel_t = build_temporal_union(data_by_time, t_ref, window)
        else:
            edge_index = data.edge_index
            if edge_time is None:
                # default zero time bias if not provided
                rel_t = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
            else:
                rel_t = edge_time

        edge_attr = self._edge_bias(edge_index, rel_t, num_nodes=N)  # [E, heads]

        h = x
        for conv, ln in zip(self.layers, self.norms):
            h_new = conv(h, edge_index, edge_attr=edge_attr)
            h = ln(h + h_new)
            h = F.gelu(h)

        z = self.final(h)  # [N, embedding_dim]
        # (Optional) normalise embeddings:
        z = F.normalize(z, p=2, dim=-1)
        return z


# -----------------------------
# Convenience wrapper
# -----------------------------
@torch.no_grad()
def make_dgt_embeddings(
    encoder: DGTEncoder,
    data: Data,
    *,
    t_ref: Optional[int] = None,
    window: Optional[Tuple[int, int]] = None,
    data_by_time: Optional[Dict[int, Data]] = None,
    edge_time: Optional[torch.Tensor] = None,
) -> Data:
    """
    Produces a new Data object with x = DGT embeddings, preserving edge_index, y, num_nodes.
    If you use dynamic graphs: pass t_ref, window, and data_by_time (mode A).
    If you already have a union graph with per-edge relative times: pass edge_time (mode B).
    """
    z = encoder(
        data,
        t_ref=t_ref,
        window=window,
        data_by_time=data_by_time,
        edge_time=edge_time
    )
    return Data(
        x=z.to(data.x.device),
        edge_index=data.edge_index,
        y=getattr(data, "y", None),
        num_nodes=data.num_nodes
    ).to(data.x.device)
