from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import numpy as np
import torch
import gc
from contextlib import contextmanager

try:
    from torch.amp import autocast as _autocast_new  # torch>=2.0 preferred API

    def _autocast_disabled(device_type: str):
        return _autocast_new(device_type=device_type, enabled=False)
except (ImportError, AttributeError):
    from torch.cuda.amp import autocast as _autocast_old  # fallback for older torch

    def _autocast_disabled(device_type: str):
        return _autocast_old(enabled=False)

def calculate_metrics(y_true, y_pred):
    precision_score_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_score_illicit = precision_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0) # illicit is class 0
    
    recall_score_weighted = recall_score(y_true, y_pred, average='weighted')
    recall_score_illicit = recall_score(y_true, y_pred, pos_label=0, average='binary') # illicit is class 0
    
    f1_score_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_score_illicit = f1_score(y_true, y_pred, pos_label=0, average='binary') # illicit is class 0
    
    metrics = {
        'precision_weighted': precision_score_weighted,
        'precision_illicit': precision_score_illicit,
        'recall_weighted': recall_score_weighted,
        'recall_illicit': recall_score_illicit,
        'f1_weighted': f1_score_weighted,
        'f1_illicit': f1_score_illicit
    }
    return metrics

import torch
import torch.nn as nn
import torch.nn.functional as F


def balanced_class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Compute inverse-frequency class weights (sum to 1) for 1-D integer labels.

    Unlabelled entries (label < 0) are ignored.
    """
    if labels.ndim != 1:
        labels = labels.view(-1)
    labels = labels.detach()
    valid = labels >= 0
    if not torch.any(valid):
        return torch.ones(num_classes, dtype=torch.float32) / float(num_classes)
    filtered = labels[valid].to(torch.long).cpu()
    counts = torch.bincount(filtered, minlength=num_classes).clamp_min(1)
    inv = (1.0 / counts.float())
    inv = inv / inv.sum()
    return inv


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            alpha_val = float(alpha)
            if not (0.0 <= alpha_val <= 1.0):
                raise ValueError("alpha float must lie in [0, 1]")
            self.alpha = torch.tensor([alpha_val, 1.0 - alpha_val], dtype=torch.float32)
        elif isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        else:
            raise TypeError("alpha must be None, float, sequence, or torch.Tensor")

        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.ndim != 1:
                raise ValueError("alpha tensor must be 1-dimensional")
            if torch.any(self.alpha < 0):
                raise ValueError("alpha tensor must be non-negative")
            if self.alpha.sum() == 0:
                raise ValueError("alpha tensor must have positive sum")
            self.alpha = self.alpha / self.alpha.sum()

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logits = inputs.float()
        targets = targets.long()
        device_type = 'cuda' if logits.is_cuda else 'cpu'
        with _autocast_disabled(device_type):
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # prob of the true class

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply alpha correctly depending on type
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha[targets]  # per-class weights
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
@contextmanager
def inference_mode_if_needed(model_name: str):
    """
    Context manager that disables gradient tracking if the model is CPU-based
    or if we are in evaluation mode.
    """
    if model_name in ["SVM", "XGB", "RF"]:
        with torch.no_grad():
            yield
    else:
        yield

def run_trial_with_cleanup(trial_func, model_name, *args, **kwargs):
    """
    Runs a trial function safely with:
      - Automatic no_grad() for CPU-based models.
      - GPU/CPU memory cleanup after each trial.
    
    Parameters
    ----------
    trial_func : callable
        The trial function to run (e.g., objective).
    model_name : str
        Name of the model (MLP, SVM, XGB, RF, GCN, GAT, GIN).
    *args, **kwargs :
        Arguments to pass to trial_func.
        
    Returns
    -------
    result : Any
        The return value of the trial function.
    """
    try:
        with inference_mode_if_needed(model_name):
            result = trial_func(*args, **kwargs)
        return result
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        
def check_study_existence(model_name, data_for_optimization):
    """
    Check if an Optuna study exists for the given model and dataset.
    
    Parameters
    ----------
    model_name : str
        Name of the model (MLP, SVM, XGB, RF, GCN, GAT, GIN).
    data_for_optimization : str
        Name of the dataset used for optimization.
        
    Returns
    -------
    exists : bool
        True if the study exists, False otherwise.
    """
    import optuna
    study_name = f'{model_name}_optimization on {data_for_optimization} dataset'
    storage_url = 'sqlite:///optimization_results.db'
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        return True
    except KeyError:
        return False
