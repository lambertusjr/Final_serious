from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
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

def calculate_metrics(y_true, y_pred, y_prob=None):
    # Standard metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    precision_score_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_score_illicit = precision_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0) # illicit is class 0
    
    recall_score_weighted = recall_score(y_true, y_pred, average='weighted')
    recall_score_illicit = recall_score(y_true, y_pred, pos_label=0, average='binary') # illicit is class 0
    
    f1_score_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_score_illicit = f1_score(y_true, y_pred, pos_label=0, average='binary') # illicit is class 0
    
    kappa = cohen_kappa_score(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision_score_weighted,
        'precision_illicit': precision_score_illicit,
        'recall_weighted': recall_score_weighted,
        'recall_illicit': recall_score_illicit,
        'f1_weighted': f1_score_weighted,
        'f1_illicit': f1_score_illicit,
        'kappa': kappa
    }

    # Metrics requiring probabilities
    if y_prob is not None:
        # Assuming y_prob has shape (N, 2) or (N,)
        # If (N, 2), we want probability of positive class (which is usually class 1, but here illicit is 0?)
        # Wait, usually illicit is the minority class. Let's check.
        # In this dataset, usually 1 is illicit?
        # The code says: pos_label=0 # illicit is class 0
        # If illicit is class 0, then we should use probability of class 0 for ROC/PR AUC if we want "illicit" specific AUC.
        # However, standard ROC AUC is usually for the "positive" class.
        # Let's calculate ROC AUC for the illicit class (class 0).
        
        if y_prob.ndim == 2:
            prob_illicit = y_prob[:, 0] # Probability of class 0
        else:
            # If 1D, assume it's prob of class 1. So prob of class 0 is 1 - prob.
            prob_illicit = 1 - y_prob
            
        # roc_auc_score handles binary labels. pos_label=0 means we treat 0 as positive.
        # But roc_auc_score usually expects y_score to be prob of positive class.
        # If we say pos_label=0, we should pass prob of class 0.
        try:
            roc_auc = roc_auc_score(y_true, prob_illicit) # By default, it might expect 1 as positive.
            # Actually, sklearn roc_auc_score: "The "greater is better" meaning of the score... y_score : array-like... Target scores."
            # If we want AUC for class 0, we treat class 0 as 1 and class 1 as 0 for the calculation, OR we just pass prob of class 0 and let it figure out?
            # No, we must be careful.
            # Let's just calculate standard ROC AUC (weighted or macro) or specifically for class 0.
            # Given the request "roc_auc", I'll provide the AUC for the illicit class since that's the focus.
            
            # To be safe with sklearn:
            # y_true has 0s and 1s.
            # If we want AUC for class 0:
            # We can flip labels: y_true_flipped = 1 - y_true (so 0 becomes 1)
            # And use prob_illicit.
            roc_auc = roc_auc_score(1 - y_true, prob_illicit)
            
            prauc = average_precision_score(1 - y_true, prob_illicit)
            
            metrics['roc_auc'] = roc_auc
            metrics['prauc'] = prauc
        except ValueError:
            # Handle cases where only one class is present in y_true
            metrics['roc_auc'] = -1
            metrics['prauc'] = -1
    else:
        metrics['roc_auc'] = -1
        metrics['prauc'] = -1
        
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
