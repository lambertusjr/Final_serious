# lstm_pipeline.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional, Tuple
from xgboost import XGBClassifier

from Helper_functions import FocalLoss, calculate_metrics, balanced_class_weights


class LSTMEncoder(nn.Module):
    """
    Minimal LSTM encoder that maps a per-node sequence x_seq ∈ R^{T×F}
    to a normalised embedding z ∈ R^{D}. Batch-first: [N, T, F] → [N, D].
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, projection_dim) if projection_dim is not None else None

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [N, T, F]
        returns: [N, D] (D = projection_dim or hidden_dim*(2 if bi else 1))
        """
        _, (h, _) = self.lstm(x_seq)  # h: [num_layers*(2 if bi else 1), N, hidden_dim]
        if self.lstm.bidirectional:
            # Concatenate last layer's forward/backward states
            h_last = torch.cat([h[-2], h[-1]], dim=-1)  # [N, 2*hidden_dim]
        else:
            h_last = h[-1]  # [N, hidden_dim]
        z = self.proj(h_last) if self.proj is not None else h_last
        return F.normalize(z, p=2, dim=-1)


def _infer_node_order(features_df, data) -> np.ndarray:
    """
    Try to recover the node order used in data.x.
    If 'data' exposes an attribute with node ids, use it; otherwise fall back
    to a deterministic order over features_df['node_id'].

    NOTE: For perfect alignment, prefer to store the node ids alongside 'data'
    during pre-processing and pass them here.
    """
    # Best case: your Data object carries an aligned node-id tensor/list
    for attr in ("node_id", "node_ids", "nodes"):
        if hasattr(data, attr):
            arr = getattr(data, attr)
            try:
                return np.asarray(arr.cpu() if hasattr(arr, "cpu") else arr)
            except Exception:
                return np.asarray(arr)

    # Fallback: stable order over unique node ids in features_df
    # (assumes pre-processing used the same deterministic order)
    return np.array(sorted(features_df["node_id"].unique()))


def _build_node_time_tensor(
    features_df,
    node_ids_ordered: np.ndarray,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build X_seq ∈ R^{N×T×F} by aligning each (node_id, time_step) row from features_df
    to the row index defined by 'node_ids_ordered' and columns ordered by time 1..T.
    Missing (node, t) are zero-filled.
    """
    # Identify feature columns
    feat_cols = [c for c in features_df.columns if c not in ("node_id", "time_step")]
    N, F = len(node_ids_ordered), len(feat_cols)
    X_seq = np.zeros((N, T, F), dtype=np.float32)

    # Efficient per-time fill; assumes time steps are 1..T (Elliptic)
    feat_index = {nid: i for i, nid in enumerate(node_ids_ordered)}
    for t in range(1, T + 1):
        ft = features_df[features_df["time_step"] == t][["node_id"] + feat_cols]
        if ft.empty:
            continue
        # Map node ids to row indices where present
        rows = ft["node_id"].map(feat_index).to_numpy()
        valid = ~np.isnan(rows)
        if not np.any(valid):
            continue
        rows = rows[valid].astype(int)
        vals = ft.loc[valid, feat_cols].to_numpy(dtype=np.float32, copy=False)
        X_seq[rows, t - 1, :] = vals

    return torch.from_numpy(X_seq).to(device)


@torch.no_grad()
def _embed_with_lstm(lstm_enc: LSTMEncoder, X_seq: torch.Tensor) -> torch.Tensor:
    lstm_enc.eval()
    return lstm_enc(X_seq)


def run_lstm_embeddings_xgb(
    *,
    features_df,                      # pandas.DataFrame with ['node_id','time_step', feature...]
    data,                             # PyG Data (for masks & labels, and device)
    train_perf_eval: torch.Tensor,    # boolean mask over nodes
    val_perf_eval: torch.Tensor,      # boolean mask over nodes
    test_perf_eval: torch.Tensor,     # boolean mask over nodes
    T: int = 49,                      # Elliptic has 49 time steps
    lstm_hidden_dim: int = 128,
    lstm_num_layers: int = 1,
    lstm_bidirectional: bool = False,
    lstm_dropout: float = 0.0,
    projection_dim: int = 256,        # final embedding size D
    warmstart_epochs: int = 20,       # 0 to disable warm-start
    warmstart_lr: float = 1e-3,
    warmstart_weight_decay: float = 5e-4,
    focal_alpha: Optional[float] = None,
    focal_gamma: float = 2.5,
    xgb_params: Optional[Dict] = None,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    End-to-end pipeline:
      1) Build [N,T,F] sequences from features_df aligned to data.x row order.
      2) (Optional) Warm-start LSTM using a linear head and focal loss on train nodes.
      3) Freeze and extract node embeddings [N,D].
      4) Train XGBoost on embeddings; return metrics and artefacts.

    Returns dict with:
      - 'embeddings' (torch.Tensor [N,D]),
      - 'xgb_model' (fitted XGBClassifier),
      - 'val_metrics' (dict),
      - 'test_metrics' (dict)
    """
    device = data.x.device
    # Ensure masks are on the same device
    train_perf_eval = train_perf_eval.to(device)
    val_perf_eval = val_perf_eval.to(device)
    test_perf_eval = test_perf_eval.to(device)

    if focal_alpha is None:
        focal_alpha = balanced_class_weights(data.y[train_perf_eval])

    # 1) Node order + sequence tensor
    node_ids_ordered = _infer_node_order(features_df, data)
    X_seq = _build_node_time_tensor(features_df, node_ids_ordered, T, device)  # [N,T,F]

    # 2) LSTM encoder (+ optional warm-start with a tiny head)
    lstm_enc = LSTMEncoder(
        input_dim=X_seq.shape[-1],
        hidden_dim=lstm_hidden_dim,
        num_layers=lstm_num_layers,
        bidirectional=lstm_bidirectional,
        dropout=lstm_dropout,
        projection_dim=projection_dim,
    ).to(device)

    if warmstart_epochs > 0:
        head = nn.Linear(projection_dim if projection_dim is not None
                         else lstm_hidden_dim * (2 if lstm_bidirectional else 1), 2).to(device)
        params = list(lstm_enc.parameters()) + list(head.parameters())
        optim = torch.optim.Adam(params, lr=warmstart_lr, weight_decay=warmstart_weight_decay)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")

        lstm_enc.train(); head.train()
        for epoch in range(warmstart_epochs):
            optim.zero_grad()
            Z = lstm_enc(X_seq)                   # [N, D]
            logits = head(Z)                      # [N, 2]
            loss = criterion(logits[train_perf_eval], data.y[train_perf_eval])
            loss.backward(); optim.step()

            # (quick sanity check on validation; no early stopping here)
            with torch.no_grad():
                lstm_enc.eval(); head.eval()
                val_logits = head(lstm_enc(X_seq))
                val_pred = val_logits[val_perf_eval].argmax(dim=-1)
                _ = calculate_metrics(
                    data.y[val_perf_eval].detach().cpu().numpy(),
                    val_pred.detach().cpu().numpy()
                )
            lstm_enc.train(); head.train()

        # We discard the head; downstream is XGB
        del head

    # 3) Freeze and extract embeddings
    with torch.no_grad():
        embeddings = _embed_with_lstm(lstm_enc, X_seq)      # [N, D]
    # Split to numpy for sklearn
    x_train = embeddings[train_perf_eval].detach().cpu().numpy()
    y_train = data.y[train_perf_eval].detach().cpu().numpy()
    x_val   = embeddings[val_perf_eval].detach().cpu().numpy()
    y_val   = data.y[val_perf_eval].detach().cpu().numpy()
    x_test  = embeddings[test_perf_eval].detach().cpu().numpy()
    y_test  = data.y[test_perf_eval].detach().cpu().numpy()

    # 4) XGBoost on embeddings
    default_xgb = dict(
        eval_metric="logloss",
        scale_pos_weight=0.108,
        learning_rate=0.1,
        max_depth=6,
        n_estimators=200,
        colsample_bytree=0.7,
        subsample=0.8,
        tree_method="hist",  # set to 'gpu_hist' if your XGBoost build has CUDA
        random_state=random_state,
    )
    if xgb_params:
        default_xgb.update(xgb_params)

    xgb_model = XGBClassifier(**default_xgb)
    xgb_model.fit(x_train, y_train)

    val_pred = xgb_model.predict(x_val)
    test_pred = xgb_model.predict(x_test)

    val_metrics = calculate_metrics(y_val, val_pred)
    test_metrics = calculate_metrics(y_test, test_pred)

    return {
        "embeddings": embeddings,         # torch.Tensor [N, D] on device
        "xgb_model": xgb_model,           # fitted model
        "val_metrics": val_metrics,       # dict from your helper
        "test_metrics": test_metrics,     # dict from your helper
    }
