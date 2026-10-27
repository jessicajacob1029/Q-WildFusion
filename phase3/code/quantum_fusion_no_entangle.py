import pennylane as qml
import torch
import torch.nn as nn

N_QUBITS = 6
N_LAYERS = 2
DEV = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(DEV, interface="torch", diff_method="parameter-shift")
def quantum_circuit_no_entangle(inputs, weights):
    # ----- Angle Encoding -----
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # ----- Variational Layers (NO entanglement) -----
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(weights[l, q, 0], wires=q)
            qml.RZ(weights[l, q, 1], wires=q)

    # ----- Measurement -----
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class QuantumFusionNoEntangle(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(N_LAYERS, N_QUBITS, 2)
        )
        self.fc = nn.Linear(N_QUBITS, N_QUBITS)

    def forward(self, x):
        outputs = []
        for i in range(x.shape[0]):
            q_out = quantum_circuit_no_entangle(x[i], self.q_weights)
            outputs.append(torch.stack(q_out))
        q_out = torch.stack(outputs).float()
        return self.fc(q_out)
