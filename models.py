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
    
class MLP(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units, dropout=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_node_features, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x = data.x  # only use node features, no graph structure
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
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
    
#%% New models
# torch>=2.0, torch-geometric>=2.4
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, BatchNorm

class GINEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_units, embedding_dim, num_layers=2, dropout=0.2):
        super(GINEncoder, self).__init__()

        if num_layers < 1:
            raise ValueError("GINEncoder requires at least one layer")

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_channels = num_node_features if layer_idx == 0 else hidden_units
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units)
            )
            self.convs.append(GINConv(mlp, train_eps=False))
            self.bns.append(BatchNorm(hidden_units))

        self.projection = nn.Linear(hidden_units, embedding_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer_idx, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if layer_idx < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.projection(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

    @torch.no_grad()
    def embed(self, data):
        self.eval()
        return self.forward(data)


