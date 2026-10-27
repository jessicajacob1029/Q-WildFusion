import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def normalize(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def plot_pca(ax, data, title):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    ax.scatter(reduced[:, 0], reduced[:, 1], s=5, alpha=0.6)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    classical = normalize(np.load("quantum_features/features_val.npy"))
    q_noent = normalize(np.load("quantum_features/features_val_quantum_no_entangle.npy"))
    q_ent_untrained = normalize(np.load("quantum_features/features_val_quantum.npy"))
    q_ent_trained = normalize(np.load("quantum_features/features_val_quantum_trained.npy"))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    plot_pca(axes[0, 0], classical, "Classical Features")
    plot_pca(axes[0, 1], q_noent, "Quantum (No Entanglement)")
    plot_pca(axes[1, 0], q_ent_untrained, "Quantum (Entangled, Untrained)")
    plot_pca(axes[1, 1], q_ent_trained, "Quantum (Entangled, Trained)")

    plt.tight_layout()
    plt.savefig("results/pca_feature_comparison.png", dpi=300)
    plt.show()

    print("PCA visualization saved to results/pca_feature_comparison.png")


if __name__ == "__main__":
    main()
