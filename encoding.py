from models import GINEncoder
from Helper_functions import FocalLoss
from Helper_functions import calculate_metrics
import copy
import torch
import torch.nn as nn
import numpy

def pre_train_GIN_encoder(data, train_perf_eval, val_perf_eval, num_classes=2, hidden_units=256, lr=0.05, weight_decay=5e-4, epochs=100, embedding_dim=128):
    device = data.x.device
    
    #Constructing GIN encoder model
    encoder = GINEncoder(
        num_node_features=data.num_node_features,
        hidden_units=hidden_units,
        embedding_dim=embedding_dim,
        num_layers=3,
        dropout=0.2
    ).to(device)
    
    #Linear head for classification. Only used to warm-start the GIN encoder.
    head = nn.Linear(embedding_dim, 2).to(device)
    
    #Setting optimizer and loss function
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=0.05, weight_decay=5e-4).to(device)
    criterion = FocalLoss(alpha=0.5, gamma=2.5, reduction='mean').to(device)
    
    #Set encoder and head to training mode
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
            val_metrics = calculate_metrics(data.y[val_perf_eval].cpu.numpy(), val_pred.cpu().numpy())
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                best_encoder_state = copy.deepcopy(encoder.state_dict())
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
            encoder.train(); head.train()
    encoder.load_state_dict(best_encoder_state)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, best_encoder_state
