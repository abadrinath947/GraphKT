import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch import nn
from torch.nn import Linear, PReLU
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

"""
class GCN(torch.nn.Module):
    def __init__(self, num_skills, hidden_dim):
        super().__init__()
        self.pre_embs = nn.Embedding(num_skills, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.prelu1 = PReLU()
        self.dropout = nn.Dropout(0.3)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.prelu2 = PReLU()
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.prelu3 = PReLU()
        self.out = Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        h1 = self.dropout(self.prelu1(self.conv1(self.pre_embs(x), edge_index, edge_weight = edge_weight)))
        h2 = self.dropout(self.prelu2(self.conv2(h1, edge_index, edge_weight = edge_weight)))
        h3 = self.prelu3(self.conv3(h2, edge_index, edge_weight = edge_weight))
        return self.out(h3)

        #return torch.cat([h1, h2, h3, h4], dim = -1)
"""

"""
class GCN(torch.nn.Module):
    def __init__(self, num_skills, hidden_dim):
        super().__init__()
        self.pre_embs = nn.Embedding(num_skills, hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim)
        self.prelu1 = PReLU()
        self.dropout = nn.Dropout(0.3)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.prelu2 = PReLU()
        self.conv3 = GATConv(hidden_dim, hidden_dim)
        self.prelu3 = PReLU()
        self.out = Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        h1 = self.dropout(self.prelu1(self.conv1(self.pre_embs(x), edge_index)))
        h2 = self.dropout(self.prelu2(self.conv2(h1, edge_index)))
        #h3 = self.prelu3(self.conv3(h2, edge_index))
        return self.out(h2)
        #return torch.cat([h1, h2, h3, h4], dim = -1)
"""

class GCN(torch.nn.Module):
    def __init__(self, num_skills, hidden_dim):
        super().__init__()
        self.pre_embs = nn.Embedding(num_skills, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.prelu1 = PReLU()
        self.dropout = nn.Dropout(0.3)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.prelu2 = PReLU()
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.prelu3 = PReLU()
        self.out = Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        h1 = self.dropout(self.prelu1(self.conv1(self.pre_embs(x), edge_index)))
        h2 = self.dropout(self.prelu2(self.conv2(h1, edge_index)))
        #h3 = self.prelu3(self.conv3(h2, edge_index))
        return self.out(h2)
        #return torch.cat([h1, h2, h3, h4], dim = -1)
