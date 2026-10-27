import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from quantum_fusion import QuantumFusionLayer

DEVICE = "cpu"
LAMBDA = 0.1
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-2


def variance_loss(x):
    return -torch.mean(torch.var(x, dim=0))


def cosine_similarity_loss(x):
    x_norm = F.normalize(x, dim=1)
    sim = torch.mm(x_norm, x_norm.t())
    i, j = torch.triu_indices(sim.shape[0], sim.shape[1], offset=1)
    return torch.mean(sim[i, j])


def main():
    feats = np.load("quantum_features/features_val.npy")
    feats = torch.tensor(feats, dtype=torch.float32, device=DEVICE)

    model = QuantumFusionLayer().to(DEVICE)
    model.train()

    optimizer = Adam(model.parameters(), lr=LR)

    print("Starting quantum circuit training...\n")

    for epoch in range(EPOCHS):
        perm = torch.randperm(feats.shape[0])
        total_loss = 0.0

        for i in range(0, feats.shape[0], BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            batch = feats[idx]

            q_out = model(batch)

            loss_var = variance_loss(q_out)
            loss_sim = cosine_similarity_loss(q_out)
            loss = loss_var + LAMBDA * loss_sim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (feats.shape[0] // BATCH_SIZE)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}")

    # Save trained quantum features
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, feats.shape[0], BATCH_SIZE):
            outputs.append(model(feats[i:i + BATCH_SIZE]).cpu().numpy())

    q_feats = np.concatenate(outputs, axis=0)
    np.save("quantum_features/features_val_quantum_trained.npy", q_feats)

    print("\nTraining complete.")
    print("Saved trained quantum features to:")
    print("quantum_features/features_val_quantum_trained.npy")


if __name__ == "__main__":
    main()
