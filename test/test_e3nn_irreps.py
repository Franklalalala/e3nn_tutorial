import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

import matplotlib.pyplot as plt

import torch
from torch import nn
from e3nn import o3


irreps = o3.Irreps("10x0o + 5x1o + 2x2e")

rot = -o3.rand_matrix()

D = irreps.D_from_matrix(rot)
print(D.size())

plt.imshow(D, cmap='bwr', vmin=-1, vmax=1)
plt.show()


