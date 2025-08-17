import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from Helper_functions import calculate_metrics

class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_units)
        self.conv2 = GCNConv(hidden_units, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
class GAT(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_units, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_units * num_heads, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x
    
class GIN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units):
        super(GIN, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units)
        )
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units)
        )
        self.conv2 = GINConv(nn2)
        self.fc = nn.Linear(hidden_units, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x
    
class ModelWrapper:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
    def train_step(self, data, mask):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data)
        loss = self.criterion(out[mask], data.y[mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def evaluate(self, data, mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            loss = self.criterion(out[mask], data.y[mask])
            pred = out.argmax(dim=1)
        metrics = calculate_metrics(data.y[mask].cpu().numpy(), pred[mask].cpu().numpy())
        return loss.item(), metrics