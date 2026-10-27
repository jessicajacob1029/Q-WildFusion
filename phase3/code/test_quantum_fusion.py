import torch
from quantum_fusion import QuantumFusionLayer

# Dummy batch of features
x = torch.randn(4, 6)

q_layer = QuantumFusionLayer()
y = q_layer(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
