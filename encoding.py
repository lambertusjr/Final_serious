from models import GINEncoder
from Helper_functions import FocalLoss
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

    criterion = FocalLoss(alpha=0.5, gamma=2.5, reduction='mean')

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
