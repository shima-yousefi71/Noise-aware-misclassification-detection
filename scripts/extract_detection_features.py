# evaluate_unsupervised_deterministic.py
import os, argparse, time
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from torch.utils.data import DataLoader
from src.data.dataset import UnlabeledDataset, ToTensor
from src.advae.advae_model import adVAE

EPS = 1e-12

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def _standardize_by_mad(arr):
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) + EPS
    return (arr - med) / mad, med, mad

def main(data_dir, model_path, output_csv,
         threshold=0.015, batch_size=1, latent_dim=128, T=5, seed=5,
         warmup=10, max_timed=None):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"[RUN] data_dir={data_dir}  output={output_csv}  T={T}  device={device}  batch={batch_size}")

    ds = UnlabeledDataset(data_dir=data_dir, transform=ToTensor())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = adVAE(input_shape=(14, 14, 512), latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rows = []

    # store per-sample times (ms)
    t_advae = []
    t_feat  = []
    t_total = []

    n_seen = 0
    n_timed = 0

    with torch.no_grad():
        for x, _, _, filenames in dl:
            x = x.to(device)

            
            if (max_timed is not None) and (n_timed >= max_timed) and (n_seen >= warmup):
               
                break

            do_time = (n_seen >= warmup)

            if do_time:
                _sync(device); t0 = time.perf_counter()

            # -------- Stage A: adVAE forwards (T times) --------
            if do_time:
                _sync(device); ta0 = time.perf_counter()

            xhats = []
            z_shifts = []
            for _ in range(T):
                x_hat, mu, logvar, z, z_t, x_t = model(x)
                # NOTE: keeping your original behavior (GPU->CPU conversion)
                xhats.append(x_hat.detach().cpu().numpy())
                z_shifts.append(torch.norm(z - z_t, dim=1).item())

            if do_time:
                _sync(device); ta1 = time.perf_counter()

            # -------- Stage B: residual/stat features --------
            if do_time:
                tb0 = time.perf_counter()

            x_hat_avg = np.mean(np.stack(xhats, axis=0), axis=0)
            latent_dist = float(np.mean(z_shifts))

            mse = F.mse_loss(
                torch.tensor(x_hat_avg, device=device).view(x.size(0), -1),
                x.view(x.size(0), -1),
                reduction="none",
            ).mean(dim=1).item()

            filename = filenames[0]
            true_label = "anomaly" if "generated" in filename else ("normal" if "original" in filename else "unknown")

            x_np = x.detach().cpu().numpy()
            resid = (x_np - x_hat_avg).astype(np.float32)

            _, resid_med, resid_mad = _standardize_by_mad(resid)

            if do_time:
                tb1 = time.perf_counter()

            if do_time:
                _sync(device); t1 = time.perf_counter()
                # store per-sample ms
                t_advae.append((ta1 - ta0) * 1000.0)
                t_feat.append((tb1 - tb0) * 1000.0)
                t_total.append((t1 - t0) * 1000.0)
                n_timed += 1

            rows.append({
                "sample": filename,
                "true_label": true_label,
                "recon_error": float(mse),
                "latent_shift": float(latent_dist),
                "resid_median": float(resid_med),
                "resid_mad": float(resid_mad),
            })

            n_seen += 1

    # ---- print timing summary (mean ± std) ----
    if len(t_total) > 0:
        print(f" Time adVAE forward (T={T}) per sample: {np.mean(t_advae):.4f} ± {np.std(t_advae):.4f} ms  (n={len(t_advae)})")
        print(f" Time Feature+stats per sample:        {np.mean(t_feat):.4f} ± {np.std(t_feat):.4f} ms  (n={len(t_feat)})")
        print(f" Time Total scoring per sample:        {np.mean(t_total):.4f} ± {np.std(t_total):.4f} ms  (n={len(t_total)})")
    else:
        print("No timed samples. Reduce warmup or ensure dataset has enough samples.")

    # ---- save CSV once ----
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"[OK] Saved {output_csv} (rows={len(rows)}, T={T})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="knn_train_original2")
    ap.add_argument("--model_path", default="models/advae_model.pt")
    ap.add_argument("--output_csv", default="results/detection_features.csv")
    ap.add_argument("--threshold", type=float, default=0.015)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--seed", type=int, default=5)

    # timing controls
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max_timed", type=int, default=None)  # e.g., 200 for one-scenario timing

    args = ap.parse_args()
    main(**vars(args))
