# train_ocsvm.py
import os, json, argparse, numpy as np, pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import joblib

# ==== Choose features  ====
FEATURE_COLS = ["recon_error", "latent_shift", "resid_median", "resid_mad"]
 #
def load_train_normals(train_csv):
    df = pd.read_csv(train_csv).copy()
    # keep originals (clean + sas); drop any generated rows if present
    is_norm = df["sample"].astype(str).str.startswith(("original", "sas"))
    df = df[is_norm]
    X = df[FEATURE_COLS].values
    return df, X

def load_mixed(mixed_csv):
    df = pd.read_csv(mixed_csv).copy()
    X = df[FEATURE_COLS].values
    # 0=normal, 1=anomaly (from filename, as in your pipeline)
    y = df["sample"].apply(lambda s: 0 if str(s).startswith(("original","sas")) else 1).values
    return df, X, y

def main(train_csv, mixed_csv, model_dir, dev_size, seed):
    os.makedirs(model_dir, exist_ok=True)

    # 1) Load TRAIN normals
    df_tr, X_tr = load_train_normals(train_csv)

    # 2) Fit robust scaler on TRAIN normals
    scaler = RobustScaler(quantile_range=(10,90))
    X_tr_s = scaler.fit_transform(X_tr)

    # 3) Load mixed, split into DEV/TEST (fixed once)
    df_mx, X_mx, y_mx = load_mixed(mixed_csv)
    X_dev, X_test, y_dev, y_test, df_dev, df_test = train_test_split(
        X_mx, y_mx, df_mx, test_size=(1.0 - (1.0-dev_size)),
        random_state=seed, stratify=y_mx
    )
    X_dev_s  = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    # 4) Grid search for (nu, gamma) on DEV
    nu_grid = [0.02, 0.04, 0.06, 0.08, 0.10]
    gam_grid = ["scale", 0.1, 0.5, 1.0, 2.0]
    best = {"model": None, "f1": -1.0, "nu": None, "gamma": None, "report": None}

    print("\n Tuning One-Class SVM on DEV (maximize F1; keep high anomaly recall)")
    for nu in nu_grid:
        for gamma in gam_grid:
            oc = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            oc.fit(X_tr_s)  # fit on normals only

            # OCSVM predict: +1=normal, -1=outlier
            y_pred_dev = oc.predict(X_dev_s)
            y_pred_dev = np.where(y_pred_dev == 1, 0, 1)  # map to 0=normal,1=anomaly

            rep = classification_report(
                y_dev, y_pred_dev, labels=[0,1],
                target_names=["normal","anomaly"], zero_division=0, output_dict=True
            )
            anom_recall = rep["anomaly"]["recall"]
            f1 = f1_score(y_dev, y_pred_dev)

            print(f"nu={nu:>4}, gamma={str(gamma):>5} | F1={f1:.4f} | anom_recall={anom_recall:.3f} | norm_recall={rep['normal']['recall']:.3f}")

            # prefer models with strong anomaly recall; break ties by F1
            score_key = (anom_recall, f1)
            best_key  = (-1.0, -1.0) if best["model"] is None else (best["report"]["anomaly"]["recall"], best["f1"])
            if (score_key > best_key):
                best = {"model": oc, "f1": f1, "nu": nu, "gamma": gamma, "report": rep}

    print(f"\n Selected nu={best['nu']} gamma={best['gamma']} | DEV F1={best['f1']:.4f}"
          f" | anom_recall={best['report']['anomaly']['recall']:.3f}")

    # 5) Lock model, evaluate once on TEST
    y_pred_test = best["model"].predict(X_test_s)
    y_pred_test = np.where(y_pred_test == 1, 0, 1)
    print("\n=== TEST Classification Report ===")
    print(classification_report(
        y_test, y_pred_test, labels=[0,1],
        target_names=["normal","anomaly"], zero_division=0
    ))

    # 6) Save artifacts
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(best["model"], os.path.join(model_dir, "ocsvm_model.pkl"))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({"feature_cols": FEATURE_COLS,
                   "nu": best["nu"], "gamma": best["gamma"]}, f, indent=2)
    print(f"\n[Saved] {model_dir}/scaler.pkl, ocsvm_model.pkl, config.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="results/dev.csv")
    ap.add_argument("--mixed_csv", default="results/test.csv")
    ap.add_argument("--model_dir", default="models/ocsvm")
    ap.add_argument("--dev_size", type=float, default=0.9, help="portion for DEV (rest is TEST)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(**vars(args))

