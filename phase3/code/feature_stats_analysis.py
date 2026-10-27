import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

def normalize(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

def load_features(path):
    return np.load(path)


def mean_l2_norm(x):
    return np.mean(np.linalg.norm(x, axis=1))


def feature_variance(x):
    return np.mean(np.var(x, axis=0))


def avg_pairwise_cosine(x, max_samples=300):
    # Subsample for speed (reviewer-safe)
    if x.shape[0] > max_samples:
        idx = np.random.choice(x.shape[0], max_samples, replace=False)
        x = x[idx]

    sims = cosine_similarity(x)
    # Take upper triangle (excluding diagonal)
    i, j = np.triu_indices_from(sims, k=1)
    return np.mean(sims[i, j])


def analyze(name, feats):
    return {
        "Feature Set": name,
        "Mean L2 Norm": mean_l2_norm(feats),
        "Feature Variance": feature_variance(feats),
        "Avg Cosine Similarity": avg_pairwise_cosine(feats),
    }


def main():
    classical = normalize(load_features("quantum_features/features_val.npy"))
    quantum_noent = normalize(load_features("quantum_features/features_val_quantum_no_entangle.npy"))
    quantum_ent = normalize(load_features("quantum_features/features_val_quantum.npy"))
    quantum_trained = normalize(load_features("quantum_features/features_val_quantum_trained.npy"))

    results = []
    results.append(analyze("Classical", classical))
    results.append(analyze("Quantum (No Entanglement)", quantum_noent))
    results.append(analyze("Quantum (Entangled, Untrained)", quantum_ent))
    results.append(analyze("Quantum (Entangled, Trained)", quantum_trained))

    print("\n=== Feature Statistics Comparison ===\n")
    for r in results:
        print(f"{r['Feature Set']}")
        print(f"  Mean L2 Norm         : {r['Mean L2 Norm']:.4f}")
        print(f"  Feature Variance     : {r['Feature Variance']:.4f}")
        print(f"  Avg Cosine Similarity: {r['Avg Cosine Similarity']:.4f}")
        print()


if __name__ == "__main__":
    main()
