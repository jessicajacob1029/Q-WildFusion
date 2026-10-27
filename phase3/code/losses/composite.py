import torch
import torch.nn.functional as F


def reliability_target(rgb, ir):
    """
    Self-supervised proxy for "which modality is more informative per
    sample", used to weakly regularize REM's learned weights.

    LLVIP is night-only (Stage 0 finding), so a day/night brightness signal
    isn't available to drive this; per the plan's Dataset Limitations note,
    thermal-contrast variation is used instead. RGB's share of the target
    tracks its own luminance/energy, thermal's share tracks its own spatial
    contrast -- the same cue story as REM's own init (rem.py), computed
    independently and detached so it acts as a fixed anchor rather than
    something REM could game by construction.
    """
    with torch.no_grad():
        luminance = rgb.flatten(1).mean(dim=1)
        contrast = ir.flatten(1).std(dim=1)
        logits = torch.stack([luminance, contrast], dim=1)
        return F.softmax(logits, dim=1)


def reliability_consistency_loss(weights, target, entropy_weight=0.01):
    """
    L_rel (advisor Step 5, composite loss). Pulls REM's live (w_rgb, w_th)
    toward the detached contrast-based `target`, which breaks collapse to a
    constant 0.5/0.5 regardless of input (the target genuinely varies with
    contrast). The subtracted entropy term breaks the opposite failure mode
    -- saturating to one-hot, always trusting a single modality -- by
    rewarding weights that stay closer to uniform. `weights`/`target`: [B, 2].
    """
    mse = F.mse_loss(weights, target)
    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=1).mean()
    return mse - entropy_weight * entropy


def quantum_regularization_loss(q_out, lambda_sim=0.1):
    """
    L_q (advisor Step 5, composite loss). Reused from train_quantum_fusion.py's
    variance_loss + cosine_similarity_loss, applied to RGVQF's raw in-detector
    quantum output (q_out: [B, N_QUBITS], pre-gate) instead of the offline
    static-feature descriptor. variance_loss discourages the circuit
    collapsing to a near-constant output regardless of input (a barren-
    plateau symptom); cosine_similarity_loss discourages redundant
    (highly-correlated) per-sample outputs.
    """
    if q_out.shape[0] < 2:
        return q_out.new_zeros(())  # variance/pairwise-cosine need >=2 samples

    variance = -torch.mean(torch.var(q_out, dim=0))

    normed = F.normalize(q_out, dim=1)
    sim = torch.mm(normed, normed.t())
    i, j = torch.triu_indices(sim.shape[0], sim.shape[1], offset=1)
    cosine = torch.mean(sim[i, j]) if i.numel() > 0 else q_out.new_zeros(())

    return variance + lambda_sim * cosine
