import os
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# Load classified results
df = pd.read_csv("results/ocsvm_predictions.csv")

def normalize_name(path_like: str) -> str:
    """Return filename without directories and without a leading 'sas_' prefix."""
    base = os.path.basename(str(path_like)).strip()
    return base[4:] if base.startswith("sas_") else base

# Build ground-truth after removing 'sas_' prefix
norm_names = df["sample"].apply(normalize_name)

def true_label(name: str) -> str:
    if name.startswith("generated"):
        return "anomaly"
    if name.startswith("original"):
        return "normal"
    return "unknown"

df["true_label_from_name"] = norm_names.apply(true_label)

# filter out 'unknown' rows (if any unusual filenames slipped in)
df_eval = df[df["true_label_from_name"] != "unknown"].copy()

y_true_str = df_eval["true_label_from_name"]
y_pred_str = df_eval["unsupervised_prediction"]

print("=== Classification Report (after stripping 'sas_' prefix) ===")
print(classification_report(y_true_str, y_pred_str))

print("=== Confusion Matrix (labels: [anomaly, normal]) ===")
cm_str = confusion_matrix(y_true_str, y_pred_str, labels=["anomaly", "normal"])
print(cm_str)
# Map to 0/1 for numeric metrics
label_map = {"normal": 0, "anomaly": 1}
y_true = y_true_str.map(label_map).values
y_pred = y_pred_str.map(label_map).values

cm = confusion_matrix(y_true, y_pred)   # default order: [0, 1] -> [normal, anomaly]
tn, fp, fn, tp = cm.ravel()

print("\n=== Confusion Matrix (numeric, normal=0, anomaly=1) ===")
print(cm)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Rates
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall for anomaly (sensitivity)
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # specificity for normal
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
balanced_acc = 0.5 * (tpr + tnr)

accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\n=== Derived Metrics ===")
print(f"Accuracy:              {accuracy:.3f}")
print(f"TPR (recall, anomaly): {tpr:.3f}")
print(f"TNR (specificity):     {tnr:.3f}")
print(f"FPR (false pos rate):  {fpr:.3f}")
print(f"FNR (false neg rate):  {fnr:.3f}")
print(f"Balanced accuracy:     {balanced_acc:.3f}")

# You can also highlight F1 just for anomaly class if you like:
f1_anom = f1_score(y_true, y_pred, pos_label=1)
f1_norm = f1_score(y_true, y_pred, pos_label=0)
print(f"\nF1 (anomaly): {f1_anom:.3f}")
print(f"F1 (normal):  {f1_norm:.3f}")

if "ocsvm_score" in df_eval.columns:
    y_score = df_eval["ocsvm_score"].values  # larger => more anomalous (see training script)
    try:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        print("\n=== Threshold-free Metrics (using ocsvm_score) ===")
        print(f"AUROC: {auroc:.3f}")
        print(f"AUPRC: {auprc:.3f}")
    except Exception as e:
        print("Could not compute AUROC/AUPRC:", e)
else:
    print("\n(No 'ocsvm_score' column found; skipping AUROC/AUPRC.)")

