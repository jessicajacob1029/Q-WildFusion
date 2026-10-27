import torch
import torch.nn as nn


class _RunningNorm(nn.Module):
    """
    Per-feature running-mean/var normalizer, functionally similar to
    BatchNorm1d(affine=False) but usable with batch size 1 in training mode.

    LLVIP's train split (6957 images) isn't divisible by the trainer's batch
    size (4), so the last batch of every epoch has size 1 — nn.BatchNorm1d
    raises ValueError on that batch in training mode. This falls back to the
    running statistics instead of erroring when there's only one sample to
    compute a batch statistic from.
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training and x.shape[0] > 1:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)
            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var
        return (x - mean) / torch.sqrt(var + self.eps)


class REM(nn.Module):
    """
    Reliability Estimation Module (advisor Step 2).

    From a pair of RGB/thermal feature maps, estimates per-modality
    reliability weights (w_rgb, w_th), softmax-normalized so they sum to 1
    per sample. These feed RGVQF's Adaptive Quantum Encoding (Stage 2) and,
    later, the reliability-consistency loss term (Stage 6).

    Cues (both computed for both modalities, so the module isn't hard-wired
    to a single signal per modality — see init below for the starting bias):
    - mean activation: a global luminance/energy proxy. LLVIP is night-only
      (Stage 0 finding), so this tracks per-scene lighting variation rather
      than a day/night signal — see Dataset Limitations in the plan.
    - spatial std: a contrast proxy. A clear thermal signature against a
      flat background gives high variance; a noisy/flat thermal read gives
      low variance.

    forward() also exposes the raw [B, 2] weights tensor (not just the two
    broadcast-shaped halves) so a trainer can log per-modality reliability
    or feed it to the reliability-consistency loss without recomputing.
    """

    def __init__(self):
        super().__init__()
        self.cue_norm = _RunningNorm(4)
        self.to_logits = nn.Linear(4, 2)

        # Bias the starting point toward the advisor's story — RGB
        # reliability tracks its own luminance/energy (cue 0), thermal
        # reliability tracks its own contrast (cue 3) — while leaving every
        # weight trainable so the composite loss (Stage 6) can move it.
        with torch.no_grad():
            self.to_logits.weight.zero_()
            self.to_logits.weight[0, 0] = 1.0  # rgb logit <- mean_rgb
            self.to_logits.weight[1, 3] = 1.0  # th logit  <- std_ir
            self.to_logits.bias.zero_()

    @staticmethod
    def _cues(feat):
        flat = feat.flatten(1)
        return flat.mean(dim=1), flat.std(dim=1)

    def forward(self, rgb, ir):
        mean_rgb, std_rgb = self._cues(rgb)
        mean_ir, std_ir = self._cues(ir)
        cues = torch.stack([mean_rgb, std_rgb, mean_ir, std_ir], dim=1)  # [B, 4]
        cues = self.cue_norm(cues)

        logits = self.to_logits(cues)  # [B, 2]
        weights = torch.softmax(logits, dim=1)  # [B, 2], sums to 1 per sample

        w_rgb = weights[:, 0].view(-1, 1, 1, 1)
        w_th = weights[:, 1].view(-1, 1, 1, 1)
        return w_rgb, w_th, weights
