import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

N_QUBITS = 8  # paper alignment (offline quantum_fusion.py circuit uses 6; reconciled up to 8 here)
N_LAYERS = 2
# B6 encoder gain: x = pi*tanh(BN(Linear(p)) / ENCODER_GAIN_K). Chosen without retraining
# (diagnostics/presubmission/b6_k_sweep.md): smallest k in {1,1.5,2,2.5,3,4} with eval-mode saturation
# (|x| > 0.9*pi) < 10% at every scale on the B6 pilot checkpoint (k=1 saturated 54/34/22% in eval).
ENCODER_GAIN_K = 2.0

_DEVICE = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(_DEVICE, interface="torch", diff_method="adjoint")
def _rgvqf_circuit(inputs, weights, entangle):
    """
    inputs:  (..., N_QUBITS) reliability-weighted, per-sample descriptor.
    weights: (N_LAYERS, N_QUBITS, 2) trainable variational parameters.
    entangle: python bool (not traced) — ring-CNOT entanglement on/off, for
              the no-entangle ablation (mirrors quantum_fusion_no_entangle.py).

    The leading batch dimension of `inputs` is handled by PennyLane's native
    parameter broadcasting rather than a per-sample Python loop — looping
    over the batch with parameter-shift (the offline quantum_fusion.py
    approach) benchmarked at ~4.9s for a 3-scale, batch-4 step; this batched
    lightning.qubit+adjoint circuit benchmarks at ~0.04s for the same shape.
    """
    for i in range(N_QUBITS):
        qml.RY(inputs[..., i], wires=i)

    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(weights[l, q, 0], wires=q)
            qml.RZ(weights[l, q, 1], wires=q)
        if entangle:
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class _BatchNorm1dB1(nn.BatchNorm1d):
    """BatchNorm1d that uses running stats for a batch of 1 in training mode
    (LLVIP train 6957 % batch 4 = 1, so the last batch of every epoch has B=1,
    where nn.BatchNorm1d raises). Otherwise identical to nn.BatchNorm1d."""

    def forward(self, x):
        if self.training and x.shape[0] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


class RGVQFFusion(nn.Module):
    """
    Reliability-Guided Variational Quantum Fusion (RGVQF).

    Adaptive Quantum Encoding (advisor Step 3): reliability weights
    (w_rgb, w_th) — normally supplied by REM (Stage 4) — scale each
    modality's features before the descriptor is built, so the encoding
    rotations emphasize whichever modality REM judges reliable for that
    sample. Until REM is wired in, weights default to 0.5/0.5 (the
    plan's specified "REM off" ablation state), reproducing an
    unweighted fusion.

    Pipeline: reliability-weight rgb/ir -> concat -> global-avg-pool ->
    adaptive-pool channel groups to N_QUBITS -> real PennyLane circuit
    (ring-CNOT entangled variational layers, ablatable) -> PauliZ
    expectations -> linear -> sigmoid channel gate -> applied to a
    (reliability-unweighted) learned spatial mix of the two modalities.
    Out channels == in channels, matching the other fusion modules so it
    drops into the P3/P4/P5 slots unchanged.
    """

    def __init__(self, channels, entangle=True, learned_encoder=False, classical_ctrl=False):
        super().__init__()
        self.channels = channels
        self.entangle = entangle
        self.last_q_out = None  # raw [B, N_QUBITS] circuit output, live (non-detached), for L_q (Stage 6)
        # learned_encoder (B6, default off): x = pi*tanh(BN(Linear(2C->8)(p))) instead of the
        # parameter-free channel-group AdaptiveAvgPool1d(2C->8), which left x nearly
        # image-independent in trained checkpoints. classical_ctrl (B7, default off): the QNode is
        # replaced by tanh(Linear(8,8)(x)) -- same encoder/gate/mix, quantum-isolation control.
        self.learned_encoder = learned_encoder
        self.classical_ctrl = classical_ctrl
        if classical_ctrl and not learned_encoder:
            raise ValueError("classical_ctrl (B7) is defined on top of learned_encoder (B6)")

        q_init = 0.01 * torch.randn(N_LAYERS, N_QUBITS, 2)
        if not classical_ctrl:  # B7 still draws q_init so every other layer inits identically to the quantum variant
            self.q_weights = nn.Parameter(q_init)
        self.to_gate = nn.Linear(N_QUBITS, channels)
        self.spatial_mix = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # identity-stable init: spatial_mix averages the two modalities, and
        # to_gate is zeroed so sigmoid(to_gate(q_out)) == 0.5 regardless of
        # q_out — together these make the module start as a plain average of
        # rgb/ir, matching QInspiredFusion's init story.
        with torch.no_grad():
            self.spatial_mix.weight.zero_()
            for i in range(channels):
                self.spatial_mix.weight[i, i, 0, 0] = 1.0
                self.spatial_mix.weight[i, i + channels, 0, 0] = 1.0
            if self.spatial_mix.bias is not None:
                self.spatial_mix.bias.zero_()

            self.to_gate.weight.zero_()
            if self.to_gate.bias is not None:
                self.to_gate.bias.zero_()

        # Created after the existing layers so flag-off construction consumes the RNG identically.
        if learned_encoder:
            self.encoder = nn.Linear(2 * channels, N_QUBITS)
            self.encoder_bn = _BatchNorm1dB1(N_QUBITS)
            self.encoder_k = ENCODER_GAIN_K
        if classical_ctrl:
            # RNG state saved/restored so every layer built after this one inits exactly as in the
            # quantum variant (B6 ON) with the same seed; classical_q's own init still depends on the seed.
            rng = torch.get_rng_state()
            self.classical_q = nn.Linear(N_QUBITS, N_QUBITS)
            torch.set_rng_state(rng)

    def _descriptor(self, rgb, ir, w_rgb, w_th):
        pooled = torch.cat([w_rgb * rgb, w_th * ir], dim=1).mean(dim=(2, 3))  # [B, 2C]
        if getattr(self, "learned_encoder", False):  # getattr: modules pickled before this flag
            k = getattr(self, "encoder_k", 1.0)  # 1.0 for modules pickled before the gain existed (B6 pilot)
            return torch.pi * torch.tanh(self.encoder_bn(self.encoder(pooled)) / k)  # [B, N_QUBITS]
        pooled = pooled.unsqueeze(1)  # [B, 1, 2C]
        descriptor = F.adaptive_avg_pool1d(pooled, N_QUBITS).squeeze(1)  # [B, N_QUBITS]
        # Bound the encoding rotation angle: raw pooled feature magnitudes are
        # unbounded, but RY expects a well-behaved angle for stable gradients.
        return torch.pi * torch.tanh(descriptor)

    def forward(self, rgb, ir, w_rgb=0.5, w_th=0.5):
        descriptor = self._descriptor(rgb, ir, w_rgb, w_th)  # [B, N_QUBITS], on whatever device rgb is on

        if getattr(self, "classical_ctrl", False):
            q_out = torch.tanh(self.classical_q(descriptor))  # [B, N_QUBITS], no QNode
        else:
            q_out = _rgvqf_circuit(descriptor.cpu(), self.q_weights.cpu(), self.entangle)
            q_out = torch.stack(q_out, dim=-1).float().to(rgb.device)  # [B, N_QUBITS]
        self.last_q_out = q_out

        gate = self.sigmoid(self.to_gate(q_out)).view(rgb.shape[0], self.channels, 1, 1)
        fused = self.spatial_mix(torch.cat([rgb, ir], dim=1))

        return fused * gate
