from __future__ import annotations

import numpy as np
import pandas as pd


def train_and_generate_vae(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_rows: int,
    seed: int = 42,
    epochs: int = 30,
    batch: int = 128,
    latent: int = 8,
    hidden: int = 64,
    learning_rate: float = 1e-3,
    kl_weight: float = 1e-3,
) -> pd.DataFrame:
    """Train a lightweight tabular VAE and sample synthetic rows."""
    import torch
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    scal = StandardScaler()

    X_num = scal.fit_transform(df[numeric_cols]) if numeric_cols else np.empty((len(df), 0))
    X_cat = enc.fit_transform(df[categorical_cols]) if categorical_cols else np.empty((len(df), 0))
    X = np.concatenate([X_num, X_cat], axis=1).astype(np.float32)
    if X.shape[1] == 0:
        raise ValueError("No features to train VAE on (empty schema).")

    class VAE(nn.Module):
        def __init__(self, d: int, latent_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mu = nn.Linear(hidden_dim, latent_dim)
            self.logvar = nn.Linear(hidden_dim, latent_dim)
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d),
            )

        def forward(self, x):
            h = self.enc(x)
            mu = self.mu(h)
            logvar = self.logvar(h)
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            z = mu + eps * std
            xr = self.dec(z)
            return xr, mu, logvar

    ds = torch.utils.data.TensorDataset(torch.tensor(X))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    vae = VAE(X.shape[1], latent_dim=latent, hidden_dim=hidden)
    opt = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    def vae_loss(x, xr, mu, logvar):
        recon = nn.MSELoss()(xr, x)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kl_weight * kld

    vae.train()
    for _ in range(epochs):
        for (xb,) in dl:
            opt.zero_grad()
            xr, mu, logvar = vae(xb)
            loss = vae_loss(xb, xr, mu, logvar)
            loss.backward()
            opt.step()

    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_rows, vae.mu.out_features)
        X_syn = vae.dec(z).cpu().numpy()

    out = pd.DataFrame(index=range(n_rows))
    idx = 0
    if X_num.shape[1] > 0:
        Xn = X_syn[:, idx : idx + X_num.shape[1]]
        Xn = scal.inverse_transform(Xn)
        for i, col in enumerate(numeric_cols):
            out[col] = Xn[:, i]
        idx += X_num.shape[1]

    if X_cat.shape[1] > 0:
        cats = enc.categories_
        start = 0
        for col, cat_vals in zip(categorical_cols, cats):
            k = len(cat_vals)
            block = X_syn[:, idx + start : idx + start + k]
            labels = np.array(cat_vals)[np.argmax(block, axis=1)]
            out[col] = labels
            start += k

    return out[df.columns]
