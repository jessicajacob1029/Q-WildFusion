import os
import numpy as np
import torch

from quantum_fusion_no_entangle import QuantumFusionNoEntangle

DEVICE = "cpu"


def main():
    in_feat_path = "quantum_features/features_val.npy"
    out_dir = "quantum_features"
    out_feat_path = os.path.join(out_dir, "features_val_quantum_no_entangle.npy")

    feats = np.load(in_feat_path)
    feats_t = torch.tensor(feats, dtype=torch.float32, device=DEVICE)

    q_layer = QuantumFusionNoEntangle().to(DEVICE)
    q_layer.eval()

    batch_size = 16
    outputs = []

    with torch.no_grad():
        for i in range(0, feats_t.shape[0], batch_size):
            batch = feats_t[i:i + batch_size]
            q_out = q_layer(batch)
            outputs.append(q_out.cpu().numpy())

    q_feats = np.concatenate(outputs, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    np.save(out_feat_path, q_feats)

    print("Quantum fusion (NO entanglement) complete.")
    print("Output shape:", q_feats.shape)
    print("Saved to:", out_feat_path)


if __name__ == "__main__":
    main()
