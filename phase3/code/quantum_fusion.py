import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

# -----------------------------
# Quantum Circuit Configuration
# -----------------------------
N_QUBITS = 6
N_LAYERS = 2
DEV = qml.device("default.qubit", wires=N_QUBITS)


# -----------------------------
# Variational Quantum Circuit
# -----------------------------
@qml.qnode(DEV, interface="torch", diff_method="parameter-shift")
def quantum_circuit(inputs, weights):
    """
    inputs: Tensor of shape (6,)
    weights: Tensor of shape (N_LAYERS, N_QUBITS, 2)
    """

    # ----- Angle Encoding -----
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # ----- Variational Layers -----
    for l in range(N_LAYERS):
        # Single-qubit rotations
        for q in range(N_QUBITS):
            qml.RX(weights[l, q, 0], wires=q)
            qml.RZ(weights[l, q, 1], wires=q)

        # Ring entanglement
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])

    # ----- Measurement -----
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# -----------------------------
# Quantum Fusion Layer (PyTorch)
# -----------------------------
class QuantumFusionLayer(nn.Module):
    def __init__(self):
        super().__init__()

        # Trainable quantum parameters
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(N_LAYERS, N_QUBITS, 2)
        )

        # Optional classical post-processing
        self.fc = nn.Linear(N_QUBITS, N_QUBITS)

    def forward(self, x):
        """
        x: Tensor of shape (B, 6)
        """
        outputs = []

        for i in range(x.shape[0]):
            q_out = quantum_circuit(x[i], self.q_weights)
            outputs.append(torch.stack(q_out))

        q_out = torch.stack(outputs).float()  # ensure float32
        return self.fc(q_out)

