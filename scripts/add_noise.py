# add_noise.py
import os, pickle, argparse, numpy as np
from glob import glob

# --- Chambers–Mallows–Stuck sampler for alpha-stable (SαS) ---
def sample_sas(size, alpha=1.5, gamma=0.05, delta=0.0, beta=0.0, rng=None):
    """
    Symmetric alpha-stable sampler (beta=0). Returns array of given size.
    alpha in (0,2]; gamma>0; delta is location.
    """
    if rng is None:
        rng = np.random.default_rng()
    U = rng.uniform(-np.pi/2, np.pi/2, size)
    W = rng.exponential(1.0, size)

    if abs(alpha - 1.0) > 1e-12:
        term = np.sin(alpha * U) / (np.cos(U) ** (1.0/alpha))
        factor = (np.cos(U - alpha * U) / W) ** ((1.0 - alpha) / alpha)
        X = term * factor
    else:
        
        X = (2/np.pi) * ( (np.pi/2 + beta*U) * np.tan(U) - beta *
                          np.log((np.pi/2)*W*np.cos(U)/(np.pi/2 + beta*U)) )
    return gamma * X + delta  # symmetric (beta≈0)

def load_feature(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        raise ValueError(f"Unsupported format in {path}")

def save_feature_dict(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"data": arr}, f)

def add_impulsive_noise(x, alpha, gamma, delta, burst_p, rng, clip=None):
    """
    Add SαS noise:
      - If burst_p is None, noise everywhere.
      - If 0 < burst_p < 1, apply Bernoulli mask so only a fraction of elements are hit.
    """
    noise = sample_sas(size=x.shape, alpha=alpha, gamma=gamma, delta=delta, beta=0.0, rng=rng)
    if burst_p is not None:
        mask = rng.random(x.shape) < burst_p
        noise = noise * mask
    y = x + noise
    if clip is not None:
        y = np.clip(y, clip[0], clip[1])
    y[~np.isfinite(y)] = 0.0
    return y

def main():
    ap = argparse.ArgumentParser(description="Contaminate a percentage of .pkl features with SαS impulsive noise and write a combined folder.")
    ap.add_argument("--input_dir", required=True, help="Folder with original .pkl files (e.g., data/layer_20)")
    ap.add_argument("--combined_dir", required=True, help="Output folder that will contain BOTH noisy and clean files together")
    ap.add_argument("--noise_fraction", type=float, default=0.3, help="Fraction of files to contaminate (0.0–1.0)")
    ap.add_argument("--alpha", type=float, default=1.5, help="SαS α in (0,2], lower=more impulsive")
    ap.add_argument("--gamma", type=float, default=0.02, help="SαS dispersion (scale) γ")
    ap.add_argument("--delta", type=float, default=0.0, help="SαS location δ")
    ap.add_argument("--burst_p", type=float, default=0.02, help="Per-element impulse probability; set -1 to add noise everywhere")
    ap.add_argument("--clip", type=float, nargs=2, default=None, help="Optional min max clip like --clip 0 6")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    assert 0.0 <= args.noise_fraction <= 1.0, "--noise_fraction must be in [0,1]"
    rng = np.random.default_rng(args.seed)
    burst_p = None if args.burst_p < 0 else float(args.burst_p)
    clip = tuple(args.clip) if args.clip is not None else None

    files = sorted([p for p in glob(os.path.join(args.input_dir, "*.pkl")) if os.path.isfile(p)])
    if not files:
        raise SystemExit(f"No .pkl files found in {args.input_dir}")

    N = len(files)
    N_noisy = int(round(args.noise_fraction * N))
    noisy_indices = set(rng.choice(N, size=N_noisy, replace=False)) if N_noisy > 0 else set()

    os.makedirs(args.combined_dir, exist_ok=True)

    print(f"Found {N} files | selecting {N_noisy} ({args.noise_fraction:.0%}) for SαS contamination")
    print(f"SαS params: α={args.alpha}, γ={args.gamma}, δ={args.delta}, burst_p={burst_p}")

    for i, in_path in enumerate(files):
        x = load_feature(in_path)
        base = os.path.basename(in_path)

        if i in noisy_indices:
            y = add_impulsive_noise(x, alpha=args.alpha, gamma=args.gamma, delta=args.delta,
                                    burst_p=burst_p, rng=rng, clip=clip)
            
            out_name = f"sas_{base}"
            out_path = os.path.join(args.combined_dir, out_name)
            save_feature_dict(out_path, y)
            tag = "noisy"
        else:
            # Copy/normalize format so everything is {"data": ...}
            out_name = base  # keep original name for clean
            out_path = os.path.join(args.combined_dir, out_name)
            save_feature_dict(out_path, x)
            tag = "clean"

        if (i + 1) % 100 == 0 or (i + 1) == N:
            print(f"[{i+1}/{N}] wrote {out_name} ({tag})")

    print(f" Combined folder ready at: {os.path.abspath(args.combined_dir)}")

if __name__ == "__main__":
    main()

