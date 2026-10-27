import pandas as pd

data = {
    "Metric": [
        "AP@50 (All)",
        "AP@50:95 (All)",
        "AP@50 (Medium)",
        "AP@50 (Large)",
        "Recall@50 (Large)"
    ],
    "Value": [
        0.791,
        0.405,
        0.479,
        0.813,
        0.942
    ]
}

df = pd.DataFrame(data)
df.to_csv("phase3/eval/final_results_table.csv", index=False)
print(df)

