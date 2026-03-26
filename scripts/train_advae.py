import os
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from src.data.dataset import UnlabeledDataset, ToTensor
from src.advae.advae_model import adVAE

# --- Hyperparameters ---
data_dir = "knn_train_original2"
batch_size = 32
latent_dim = 128
input_shape = (14, 14, 512)
epochs = 200
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- DataLoader ---
dataset = UnlabeledDataset(data_dir=data_dir, transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Model and Optimizer ---
model = adVAE(input_shape=input_shape, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# =====================================================================
#  NEW: helper to compute reconstruction MSE + MAD (per batch)
# =====================================================================
def recon_stats(x_hat, x):
    """
    Compute reconstruction MSE and MAD (robust) between x_hat and x.

    Returns:
      mse_rec  : scalar, mean MSE over batch
      mad_rec  : scalar, mean MAD over batch
      mse_per  : (B,), per-sample MSE (if you ever want it)
      mad_per  : (B,), per-sample MAD
    """
    B = x.size(0)

    x_flat     = x.view(B, -1)
    x_hat_flat = x_hat.view(B, -1)

    resid   = x_hat_flat - x_flat          # (B, D)
    abs_res = resid.abs()

    # MSE per sample
    mse_per = (resid ** 2).mean(dim=1)     # (B,)
    mse_rec = mse_per.mean()               # scalar

    # MAD per sample: median(|residual|)
    mad_per = abs_res.median(dim=1).values # (B,)
    mad_rec = mad_per.mean()               # scalar

    return mse_rec, mad_rec, mse_per, mad_per

# =====================================================================
#  MODIFIED: adVAE loss with robust MAD term, but KEEPING MSE & LS
# =====================================================================
def advae_loss(
    x, x_hat, mu, logvar, z, z_t, x_t,
    beta=1.0,
    gamma=1.0,
    mad_weight=1.0,
    mse_weight=0.0,  # set >0 if you want some MSE influence in training
):
    """
    Returns:
      total_loss        : scalar for backprop
      mse_rec           : avg reconstruction MSE          (indicator 1)
      latent_shift_mean : avg ||z_t - z||^2              (indicator 2)
      mad_rec           : avg MAD of residuals           (indicator 3)
      adv_mad           : MAD for adversarial recon x_t
    """

    # Make sure shapes are consistent for recon
    # (B, ...) from model; we only flatten inside recon_stats if needed
    # x, x_hat, x_t : (B, C, H, W) or (B, D, ...)
    # z, z_t       : (B, latent_dim)

    # --- Reconstruction stats on benign branch ---
    mse_rec, mad_rec, _, _ = recon_stats(x_hat, x)

    # Robust reconstruction part of the loss
    recon_loss = mad_weight * mad_rec + mse_weight * mse_rec

    # --- Latent shift (feature you keep) ---
    # per-sample ||z_t - z||^2
    ls_per = ((z_t - z) ** 2).mean(dim=1)   # (B,)
    latent_shift_mean = ls_per.mean()       # scalar

    # --- KL loss (standard VAE term) ---
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # --- Adversarial reconstruction branch (also robust via MAD) ---
    B = x.size(0)
    x_flat   = x.view(B, -1)
    x_t_flat = x_t.view(B, -1)

    resid_t = (x_t_flat - x_flat).abs()
    adv_mad = resid_t.median(dim=1).values.mean()  # scalar

    # Total loss: recon (MAD-dominated) + KL + adversarial terms
    total_loss = recon_loss + beta * kl_loss + gamma * (latent_shift_mean + adv_mad)

    return total_loss, mse_rec, latent_shift_mean, mad_rec, adv_mad

# =====================================================================
#  Training Loop (slightly extended logging)
# =====================================================================
model.train()
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    total_mse  = 0.0
    total_ls   = 0.0
    total_mad  = 0.0

    for batch in dataloader:
        x = batch[0].to(device)

        optimizer.zero_grad()
        x_hat, mu, logvar, z, z_t, x_t = model(x)

        loss, mse_rec, ls_mean, mad_rec, adv_mad = advae_loss(
            x, x_hat, mu, logvar, z, z_t, x_t,
            beta=1.0,
            gamma=1.0,
            mad_weight=1.0,   # <-- main driver: robust to SαS noise
            mse_weight=0.0    # <-- you can try 0.1 later if needed
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse  += mse_rec.item()
        total_ls   += ls_mean.item()
        total_mad  += mad_rec.item()

    n_batches = len(dataloader)
    print(
        f"Epoch {epoch:03d} | "
        f"Loss {total_loss/n_batches:.4f} | "
        f"MSE {total_mse/n_batches:.4f} | "
        f"LS {total_ls/n_batches:.4f} | "
        f"MAD {total_mad/n_batches:.4f}"
    )

# --- Save Model ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/advae_model.pt")

