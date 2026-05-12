"""
models/cmb_model.py
───────────────────
Full CMB reconstruction architecture from the paper.

Components
──────────
  (i)   MultiFreqEncoder          — multi-frequency tensor encoder
  (ii)  WaveletDecompose          — differentiable DWT via F.conv2d (pure PyTorch)
  (iii) SpectralAttention         — scale-aware spectral attention
  (iv)  ForegroundLatentEncoder   — physics-informed latent z
  (v)   UNetDenoiser              — conditional DDPM in wavelet space
  (vi)  PowerSpectrumLoss         — differentiable 2-D power spectrum match
  (vii) CMBDiffusionModel         — assembles all of the above

No healpy, no astropy-healpix, no PyWavelets. Pure PyTorch throughout.
All transforms are differentiable — gradients flow end-to-end.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Multi-Frequency Tensor Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """GroupNorm residual block."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, dim), nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim), nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MultiFreqEncoder(nn.Module):
    """
    Encodes X ∈ R^{C×H×W} → h ∈ R^{D×H/4×W/4}.
    h = f_θ(X)
    """
    def __init__(self, in_channels: int = 3, base_dim: int = 64, depth: int = 3):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        self.res  = nn.Sequential(*[ResBlock(base_dim) for _ in range(depth)])
        self.dn1  = nn.Conv2d(base_dim,     base_dim * 2, 4, stride=2, padding=1)
        self.dn2  = nn.Conv2d(base_dim * 2, base_dim * 2, 4, stride=2, padding=1)
        self.out_channels = base_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.res(self.stem(x))
        h = F.silu(self.dn1(h))
        return F.silu(self.dn2(h))


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Wavelet Decomposition  (fully differentiable, pure PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

class WaveletDecompose(nn.Module):
    """
    Differentiable 2-D Haar DWT via depthwise F.conv2d.

    X (N, C, H, W) → (N, 4C, H//2, W//2)
    Sub-bands per input channel: [LL, LH, HL, HH]

    Why Haar?
    ─────────
    The Haar filters are exact rational numbers, trivially expressed as
    PyTorch tensors.  The transform is a strided depthwise convolution with
    circular padding — fully in-graph, no numpy, no detach.
    Gradients flow back through the transform to the input and to any
    downstream module that uses the sub-band features.

    Inverse (for _from_wavelet):
    ────────────────────────────
    We only need LL (the low-pass approximation) for reconstruction.
    The transpose conv of the LL filter with stride 2 upsamples it back
    to the original spatial size, acting as the inverse approximation.
    """

    # Haar analysis filters (length-2, orthonormal):
    #   Lo =  [1, 1] / sqrt(2)   (low-pass)
    #   Hi =  [1,-1] / sqrt(2)   (high-pass)
    # 2-D separable: LL, LH, HL, HH
    _LO =  0.7071067811865476   # 1/√2
    _HI =  0.7071067811865476

    def __init__(self):
        super().__init__()
        # Build the four 2-D Haar analysis kernels, shape (1,1,2,2)
        # stored as non-trainable buffers so they move with .to(device)
        s = self._LO
        kernels = torch.tensor([
            [[ s,  s], [ s,  s]],   # LL  (low × low)
            [[ s,  s], [-s, -s]],   # LH  (low × high) — note: row direction
            [[ s, -s], [ s, -s]],   # HL  (high × low)
            [[ s, -s], [-s,  s]],   # HH  (high × high)
        ]) * s                       # already normalised above, *s for 2-D
        # Shape: (4, 1, 2, 2)
        self.register_buffer("kernels", kernels.unsqueeze(1))

        # Synthesis (inverse) kernel for LL only — same as analysis LL
        self.register_buffer("inv_kernel", kernels[0].unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, C, H, W)  →  out : (N, 4C, H//2, W//2)

        Applies the four Haar kernels to every input channel independently
        using a grouped depthwise convolution with stride 2 and circular
        (periodic) padding of 0 — Haar is length-2 so no padding needed
        when H,W are even.
        """
        N, C, H, W = x.shape
        # Expand kernels to cover all C channels: (4C, 1, 2, 2)
        k = self.kernels.repeat(C, 1, 1, 1)      # (4C, 1, 2, 2)
        # Treat each (channel × sub-band) as its own group
        # We need to interleave: for each input channel c, produce 4 bands.
        # Reshape x → (N, C, H, W), tile kernels appropriately.
        # Easiest: process all channels at once with groups=C.
        # kernels shape needed: (4C, 1, 2, 2) with groups=C → each group
        # gets 1 input channel and 4 output filters.
        # PyTorch grouped conv: out_channels must be divisible by groups,
        # in_channels  must equal groups (depthwise).
        # So we do it per-band instead — 4 convolutions each producing C maps.
        bands = []
        for i in range(4):
            ki = self.kernels[i].expand(C, 1, 2, 2)   # (C, 1, 2, 2)
            bi = F.conv2d(x, ki, stride=2,
                          padding=0, groups=C)          # (N, C, H//2, W//2)
            bands.append(bi)
        return torch.cat(bands, dim=1)                 # (N, 4C, H//2, W//2)

    def inverse_ll(self, ll: torch.Tensor) -> torch.Tensor:
        """
        Upsample the LL sub-band back to (H, W) using the transpose conv.
        ll : (N, 1, H//2, W//2)  →  (N, 1, H, W)
        """
        return F.conv_transpose2d(ll, self.inv_kernel, stride=2, padding=0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Scale-Aware Spectral Attention
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralAttention(nn.Module):
    """
    A_{ν,ℓ} = Softmax(g_θ(X_{ν,ℓ}))
    Learns which wavelet sub-band × frequency combination is most informative.
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(in_channels, mid), nn.SiLU(),
            nn.Linear(mid, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s    = self.pool(x).flatten(1)
        attn = torch.softmax(self.fc(s), dim=-1)
        return x * attn[:, :, None, None]


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Latent Foreground Physics Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class ForegroundLatentEncoder(nn.Module):
    """
    Predicts z = (β_sync, β_dust, T_dust) from encoder features.
    A Gaussian prior KL enforces physically plausible foreground parameters.
    """
    def __init__(self, encoder_channels: int, latent_dim: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(encoder_channels, 64), nn.SiLU(),
            nn.Linear(64, latent_dim),
        )
        self.register_buffer("prior_mu",  torch.tensor([-3.1,  1.6, 20.0]))
        self.register_buffer("prior_std", torch.tensor([ 0.15, 0.15,  2.0]))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_pred = self.head(self.pool(h).flatten(1))
        kl     = 0.5 * ((z_pred - self.prior_mu) ** 2
                         / self.prior_std ** 2).mean()
        return z_pred, kl


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Sinusoidal Time Embedding
# ═══════════════════════════════════════════════════════════════════════════════

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (N,) int  →  (N, dim) sinusoidal embedding."""
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Conditional U-Net Denoiser
# ═══════════════════════════════════════════════════════════════════════════════

class CondResBlock(nn.Module):
    """Residual block conditioned on timestep embedding and encoder features."""
    def __init__(self, dim: int, t_dim: int, h_dim: int):
        super().__init__()
        self.n1    = nn.GroupNorm(8, dim)
        self.c1    = nn.Conv2d(dim, dim, 3, padding=1)
        self.n2    = nn.GroupNorm(8, dim)
        self.c2    = nn.Conv2d(dim, dim, 3, padding=1)
        self.t_pr  = nn.Linear(t_dim, dim)
        self.h_pr  = nn.Conv2d(h_dim, dim, 1)

    def forward(self, x: torch.Tensor,
                t_emb: torch.Tensor,
                h: torch.Tensor) -> torch.Tensor:
        t_c = self.t_pr(t_emb)[:, :, None, None]
        h_c = F.interpolate(self.h_pr(h), size=x.shape[2:],
                            mode="bilinear", align_corners=False)
        x2  = self.c1(F.silu(self.n1(x))) + t_c + h_c
        return x + self.c2(F.silu(self.n2(x2)))


class UNetDenoiser(nn.Module):
    """
    Predicts noise ε in wavelet space, conditioned on:
        h  — encoder features
        t  — diffusion timestep
    """
    def __init__(self, wavelet_channels: int, encoder_channels: int,
                 model_dim: int = 64, t_dim: int = 128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2), nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.in_conv  = nn.Conv2d(wavelet_channels, model_dim, 3, padding=1)
        self.blocks   = nn.ModuleList([
            CondResBlock(model_dim, t_dim, encoder_channels),
            CondResBlock(model_dim, t_dim, encoder_channels),
        ])
        self.out_conv = nn.Conv2d(model_dim, wavelet_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor,
                t: torch.Tensor,
                h: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.t_dim))
        x     = self.in_conv(x_t)
        for blk in self.blocks:
            x = blk(x, t_emb, h)
        return self.out_conv(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Differentiable Power Spectrum Loss
# ═══════════════════════════════════════════════════════════════════════════════

def power_spectrum_1d(img: torch.Tensor) -> torch.Tensor:
    """
    Azimuthally averaged 2-D power spectrum.
    img : (N, H, W)  →  Pₖ : (N, H//2)
    """
    N, H, W = img.shape
    fft     = torch.fft.rfft2(img)
    power   = fft.abs() ** 2 / (H * W)

    ky = torch.fft.fftfreq (H, device=img.device)[:, None]
    kx = torch.fft.rfftfreq(W, device=img.device)[None, :]
    kr = (ky**2 + kx**2).sqrt().flatten()

    n_bins  = H // 2
    k_max   = kr.max()
    bin_idx = (kr * n_bins / k_max).long().clamp(0, n_bins - 1)

    Pk      = torch.zeros(N, n_bins, device=img.device)
    p_flat  = power.reshape(N, -1)
    Pk.scatter_add_(1, bin_idx.unsqueeze(0).expand(N, -1), p_flat)
    return Pk


class PowerSpectrumLoss(nn.Module):
    """
    L_PS = Σ_ℓ |C_ℓ^pred/C̄_ℓ^true − 1|²

    Normalised by the target power spectrum so the loss is dimensionless and
    O(1) regardless of map units (μK, normalised, etc.).  Without this the
    raw MSE on C_ℓ values can be O(10⁴–10⁶) at initialisation, completely
    swamping the diffusion loss.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        Pk_pred = power_spectrum_1d(pred)
        Pk_tgt  = power_spectrum_1d(target)
        # Normalise per-sample by mean target power; clamp to avoid /0
        norm = Pk_tgt.detach().mean(dim=1, keepdim=True).clamp(min=1e-8)
        return F.mse_loss(Pk_pred / norm, Pk_tgt / norm)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Full CMB Diffusion Model
# ═══════════════════════════════════════════════════════════════════════════════

class CMBDiffusionModel(nn.Module):
    """
    Probabilistic CMB reconstruction via wavelet-domain conditional diffusion.

    Training:  model(X, y, z_true) → loss dict
    Inference: model.sample(X, n_samples) → (S, N, P, P) tensor of samples
    """

    def __init__(
        self,
        in_channels:   int   = 3,
        patch_size:    int   = 64,
        base_dim:      int   = 64,
        encoder_depth: int   = 3,
        T_diffusion:   int   = 50,
        lambda_ps:  float = 0.10,
        lambda_z:   float = 0.01,
    ):
        super().__init__()
        self.T           = T_diffusion
        self.lambda_ps = lambda_ps
        self.lambda_z  = lambda_z

        self.encoder    = MultiFreqEncoder(in_channels, base_dim, depth=encoder_depth)
        self.wavelet    = WaveletDecompose()
        self.spec_attn  = SpectralAttention(4 * in_channels)
        # Projects attention-weighted wavelet features into encoder space
        self.attn_proj  = nn.Conv2d(4 * in_channels, self.encoder.out_channels, 1)
        self.fg_encoder = ForegroundLatentEncoder(self.encoder.out_channels)
        self.denoiser   = UNetDenoiser(
            wavelet_channels = 4,
            encoder_channels = self.encoder.out_channels,
            model_dim        = base_dim,
        )
        self.ps_loss = PowerSpectrumLoss()

        # Linear β noise schedule
        betas     = torch.linspace(1e-4, 0.02, T_diffusion)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    # ── Wavelet helpers ───────────────────────────────────────────────────────

    def _to_wavelet(self, y: torch.Tensor) -> torch.Tensor:
        """(N,P,P) → (N,4,P',P')"""
        return self.wavelet(y.unsqueeze(1))

    def _from_wavelet(self, w: torch.Tensor) -> torch.Tensor:
        """(N,4,P//2,P//2) -> (N,P,P) via differentiable Haar inverse (LL band)."""
        return self.wavelet.inverse_ll(w[:, :1]).squeeze(1)

    # ── Diffusion helpers ─────────────────────────────────────────────────────

    def _q_sample(self, x0: torch.Tensor,
                  t: torch.Tensor,
                  noise: torch.Tensor | None = None):
        """Forward noising: xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(self, X: torch.Tensor,
                y: torch.Tensor,
                z_true: torch.Tensor | None = None) -> dict:
        """
        X      : (N, C, P, P)  multi-freq observations
        y      : (N, P, P)     true CMB
        z_true : (N, 3) | None foreground labels (simulations only)
        """
        N, C, P, _ = X.shape
        device      = X.device

        # Encode multi-frequency input
        h    = self.encoder(X)

        # Wavelet decompose → spectral attention → residual-add into h
        w_X  = self.wavelet(X)
        w_Xa = self.spec_attn(w_X)                          # (N, 4C, H', W')
        h_a  = F.interpolate(self.attn_proj(w_Xa),
                             size=h.shape[2:],
                             mode="bilinear", align_corners=False)
        h    = h + h_a                                       # attention-conditioned

        z_pred, kl_loss = self.fg_encoder(h)

        # Forward diffusion on wavelet-domain target
        w_y        = self._to_wavelet(y)
        t          = torch.randint(0, self.T, (N,), device=device)
        w_yt, eps  = self._q_sample(w_y, t)
        eps_pred   = self.denoiser(w_yt, t, h)

        # Losses
        L_diffusion = F.mse_loss(eps_pred, eps)

        ab        = self.alpha_bar[t][:, None, None, None]
        w_pred    = (w_yt - (1 - ab).sqrt() * eps_pred) / ab.sqrt()
        y_pred    = self._from_wavelet(w_pred)
        L_ps      = self.ps_loss(y_pred, y)

        # Cross-frequency consistency removed:
        # Running self.encoder C extra times per step quadruples memory usage
        # and contributed L_freq=0.000 throughout run1 — not worth the cost.

        L_z = (F.mse_loss(z_pred, z_true) if z_true is not None else 0.0)               + self.lambda_z * kl_loss

        loss = (L_diffusion
                + self.lambda_ps * L_ps
                + self.lambda_z  * L_z)

        return {
            "loss":        loss,
            "L_diffusion": L_diffusion.item(),
            "L_ps":        L_ps.item(),
            "L_freq":      0.0,
            "L_z":         L_z.item() if isinstance(L_z, torch.Tensor) else float(L_z),
        }

    # ── Inference (DDPM reverse) ──────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, X: torch.Tensor, n_samples: int = 1, temperature: float = 1.0) -> torch.Tensor:
        """
        Draw n_samples realisations of T_CMB given multi-freq observation X.
        Returns (n_samples, N, P, P).
        """
        N, C, P, _ = X.shape
        device      = X.device

        # Encode — mirror training conditioning exactly
        h    = self.encoder(X)
        w_X  = self.wavelet(X)
        w_Xa = self.spec_attn(w_X)
        h_a  = F.interpolate(self.attn_proj(w_Xa),
                             size=h.shape[2:],
                             mode="bilinear", align_corners=False)
        h    = h + h_a

        # Haar DWT halves spatial dims exactly
        Pw = P // 2

        samples = []
        for _ in range(n_samples):
            w_t = torch.randn(N, 4, Pw, Pw, device=device)

            for step in reversed(range(self.T)):
                t_b      = torch.full((N,), step, device=device, dtype=torch.long)
                eps_pred = self.denoiser(w_t, t_b, h)

                ab    = self.alpha_bar[step]
                alpha = self.alphas[step]
                beta  = self.betas[step]

                if step > 0:
                    ab_prev    = self.alpha_bar[step - 1]
                    # DDPM posterior variance (Ho et al. 2020, Eq. 7)
                    beta_tilde = (1.0 - ab_prev) / (1.0 - ab) * beta

                    # DDPM reverse mean (Ho et al. 2020, Eq. 11)
                    # mu = 1/sqrt(a_t) * (x_t - b_t/sqrt(1-ab_t) * eps)
                    w_t = (1.0 / alpha.sqrt()) * (
                        w_t - beta / (1.0 - ab).sqrt() * eps_pred
                    ) + beta_tilde.sqrt() * temperature * torch.randn_like(w_t)
                else:
                    # Final step: return mean, no noise
                    w_t = (w_t - (1.0 - ab).sqrt() * eps_pred) / ab.sqrt()
                    w_t = w_t.clamp(-5.0, 5.0)

            samples.append(self._from_wavelet(w_t))

        return torch.stack(samples)   # (n_samples, N, P, P)