#scripts/split_dataset.py
import os, hashlib, pandas as pd

INPUT_CSV = "results/detection_features.csv"
DEV_CSV = "results/dev.csv"
TEST_CSV = "results/test.csv"

def base_no_sas(s: str) -> str:
    b = os.path.basename(str(s))
    return b[4:] if b.startswith("sas_") else b  # strip leading "sas_"

def is_normal(s: str) -> bool:
    b = base_no_sas(s)
    return b.startswith("original")

def bucket(name: str, mod=5) -> int:
    h = hashlib.md5(base_no_sas(name).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod  # stable 0..4

df = pd.read_csv(INPUT_CSV).copy()
b = df["sample"].apply(bucket)

df_dev  = df[b != 0].copy()  # ~80%
df_test = df[b == 0].copy()  # ~20%
os.makedirs("results", exist_ok=True)
df_dev.to_csv(DEV_CSV, index=False)
df_test.to_csv(TEST_CSV, index=False)

print("DEV counts:",
      (df_dev["sample"].apply(is_normal)).sum(), "normal /",
      (~df_dev["sample"].apply(is_normal)).sum(), "anomaly")
print("TEST counts:",
      (df_test["sample"].apply(is_normal)).sum(), "normal /",
      (~df_test["sample"].apply(is_normal)).sum(), "anomaly")

