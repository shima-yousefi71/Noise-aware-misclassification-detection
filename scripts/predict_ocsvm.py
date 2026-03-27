# scripts/predict_ocsvm.py
import os, json, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

MODEL_DIR = "models/ocsvm"

def classify_new(csv_path, output_csv="results/ocsvm_predictions.csv"):
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)
    feat = cfg["feature_cols"]

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model  = joblib.load(os.path.join(MODEL_DIR, "ocsvm_model.pkl"))

    df = pd.read_csv(csv_path).copy()
    X  = df[feat].values
    _ = scaler.transform(X[:1])
    _ = model.predict(scaler.transform(X[:1]))
    _ = model.decision_function(scaler.transform(X[:1]))

    t0 = time.perf_counter()
    Xs = scaler.transform(X)

    # OCSVM: +1 normal, -1 outlier
    yhat = model.predict(Xs)
    ocsvm_score = -model.decision_function(Xs)
    t1 = time.perf_counter()

    infer_total_ms = (t1 - t0) * 1000.0
    infer_per_sample_ms = infer_total_ms / max(len(X), 1)

    print(f" OCSVM inference total: {infer_total_ms:.4f} ms")
    print(f" OCSVM latency per sample: {infer_per_sample_ms:.6f} ms")

    df["unsupervised_prediction"] = np.where(yhat == 1, "normal", "anomaly")
    df["ocsvm_score"] = ocsvm_score


    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    # same scatter you already use
    if {"recon_error","latent_shift"}.issubset(df.columns):
        plt.figure(figsize=(8,6))
        sns.scatterplot(
            data=df, x="recon_error", y="latent_shift",
            hue="unsupervised_prediction", alpha=0.7,
            palette={"normal":"green","anomaly":"red"}
        )
        plt.title("One-Class SVM (RBF) — RE vs LS")
        plt.tight_layout()
        plot_path = "results/ocsvm_prediction_plot.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Plot saved to {plot_path}")
    return df

if __name__ == "__main__":
    classify_new("results/test.csv")

