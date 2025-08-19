from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np

def calculate_metrics(y_true, y_pred):
    precision_score_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_score_illicit = precision_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)
    
    recall_score_weighted = recall_score(y_true, y_pred, average='weighted')
    recall_score_illicit = recall_score(y_true, y_pred, pos_label=0, average='binary')
    
    f1_score_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_score_illicit = f1_score(y_true, y_pred, pos_label=0, average='binary')
    
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
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha  # keep as float
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # prob of the true class

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply alpha correctly depending on type
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):  
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                at = self.alpha[targets]  # per-class weights
                focal_loss = at * focal_loss
            else:  
                focal_loss = self.alpha * focal_loss  # float case

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss