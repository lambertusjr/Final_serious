import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv

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