import os
import numpy as np
import torch

from quantum_fusion import QuantumFusionLayer

DEVICE = "cpu"  # quantum simulator runs on CPU


def main():
    # ---- Paths ----
    in_feat_path = "quantum_features/features_val.npy"
    out_dir = "quantum_features"
    out_feat_path = os.path.join(out_dir, "features_val_quantum.npy")

    # ---- Load features ----
    feats = np.load(in_feat_path)              # (N, 6)
    feats_t = torch.tensor(feats, dtype=torch.float32, device=DEVICE)

    # ---- Load quantum fusion layer ----
    q_layer = QuantumFusionLayer().to(DEVICE)
    q_layer.eval()

    # ---- Run quantum fusion (batch-wise for stability) ----
    batch_size = 16  # small batch = stable + fast enough
    outputs = []

    with torch.no_grad():
        for i in range(0, feats_t.shape[0], batch_size):
            batch = feats_t[i:i + batch_size]
            q_out = q_layer(batch)              # (B, 6)
            outputs.append(q_out.cpu().numpy())

    q_feats = np.concatenate(outputs, axis=0)

    # ---- Save outputs ----
    os.makedirs(out_dir, exist_ok=True)
    np.save(out_feat_path, q_feats)

    print("Quantum fusion complete.")
    print("Input features shape :", feats.shape)
    print("Output features shape:", q_feats.shape)
    print("Saved to:", out_feat_path)


if __name__ == "__main__":
    main()
