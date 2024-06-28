from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import nn as enn
from e3nn.math import soft_one_hot_linspace
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

import matplotlib.pyplot as plt

import torch
from torch import nn
from e3nn import o3




irreps_input = o3.Irreps("10x0e + 10x1e")
irreps_output = o3.Irreps("20x0e + 10x1e")

num_nodes = 100
pos = torch.randn(num_nodes, 3)  # random node positions

# create edges
max_radius = 1.8
edge_src, edge_dst = radius_graph(pos, max_radius, max_num_neighbors=num_nodes - 1)

edge_vec = pos[edge_dst] - pos[edge_src]

# compute z
num_neighbors = len(edge_src) / num_nodes

f_in = irreps_input.randn(num_nodes, -1)
print(f_in.size())

irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
print(irreps_sh)

sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='component')
print(edge_vec.size(), sh.size())

tp = o3.FullyConnectedTensorProduct(irreps_in1=irreps_input, irreps_in2=irreps_sh, irreps_out=irreps_output, shared_weights=False)

print(tp)
print(tp.instructions)
print(o3.FullTensorProduct(irreps_in1=irreps_input, irreps_in2=irreps_sh))

num_basis = 10

edge_length_embedding = soft_one_hot_linspace(
    edge_vec.norm(dim=1),
    start=0.0,
    end=max_radius,
    number=num_basis,
    basis='smooth_finite',
    cutoff=True,
)
edge_length_embedding = edge_length_embedding.mul(num_basis**0.5)

print(edge_vec.size(), edge_length_embedding.size())

fc = enn.FullyConnectedNet([num_basis, 16, tp.weight_numel], torch.relu)
weight = fc(edge_length_embedding)

print(weight.shape)

print('\nParameters of network h:')
for i in fc.parameters():
    print(i.size())

summand = tp(f_in[edge_src], sh, weight)

print('\n', summand.size())

f_out = scatter(summand, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors ** 0.5)

print(f_out.size())

