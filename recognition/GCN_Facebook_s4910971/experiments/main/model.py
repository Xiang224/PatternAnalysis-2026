"""Handwritten GCN with separate feature-transform and aggregation modules."""

import torch
import torch.nn.functional as F
from torch import nn


def normalise_edges(edge_index, num_nodes, dtype):
    """Keep the connections, add one self-loop per node and normalise by degree."""
    source, target = edge_index
    keep = source != target
    nodes = torch.arange(num_nodes, device=edge_index.device)
    source = torch.cat((source[keep], nodes))
    target = torch.cat((target[keep], nodes))

    degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
    degree.index_add_(0, target, torch.ones_like(target, dtype=dtype))
    inverse_sqrt_degree = degree.pow(-0.5)
    coefficient = inverse_sqrt_degree[source] * inverse_sqrt_degree[target]
    return source, target, coefficient


class NeighbourAggregation(nn.Module):
    """Sum transformed neighbour features using fixed degree coefficients."""

    def forward(self, x, source, target, coefficient):
        messages = x[source] * coefficient.unsqueeze(1)
        output = torch.zeros_like(x)
        output.index_add_(0, target, messages)
        return output


class GraphConvolution(nn.Module):
    """One graph convolution: feature transformation, aggregation, then bias."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim, bias=False)
        self.aggregate = NeighbourAggregation()
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, source, target, coefficient):
        x = self.lin(x)
        x = self.aggregate(x, source, target, coefficient)
        return x + self.bias


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, dropout):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(input_dim, hidden_dim))
        for _ in range(depth - 2):
            self.layers.append(GraphConvolution(hidden_dim, hidden_dim))
        self.layers.append(GraphConvolution(hidden_dim, 4))

    def forward(self, x, edge_index):
        source, target, coefficient = normalise_edges(edge_index, x.size(0), x.dtype)
        for layer in self.layers[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(layer(x, source, target, coefficient))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, source, target, coefficient)
