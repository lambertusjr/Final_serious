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
    embedding_dim=128
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
            if val_metrics['f1_illicit'] > best_val_f1:
                best_val_f1 = val_metrics['f1_illicit']
                best_encoder_state = copy.deepcopy(encoder.state_dict())
            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Loss: {loss.item():.4f}, "
                f"Val illicit recall: {val_metrics['recall_illicit']:.4f}, "
                f"Val illicit F1: {val_metrics['f1_illicit']:.4f}"
            )
            encoder.train(); head.train()

    encoder.load_state_dict(best_encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder, best_encoder_state
