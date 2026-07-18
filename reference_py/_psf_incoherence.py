"""N2 incoherence probe v2 — alias PSF relative to the fully-sampled reference.

PSF_w    = A_w^H D_w A_w delta   (one nav window, peak-normalized)
PSF_full = A_832^H D A_832 delta (all 832 interleaves = full Thomson set)
alias    = PSF_w - PSF_full      (complex difference; the shared aperture/Gibbs
                                  ring cancels -> pure undersampling alias field)

Statistics use the proper noise scale: sigma = sqrt(E|x|^2 / 2) of the complex
field (Rayleigh magnitudes), expected noise-like max = sigma*sqrt(2 ln n).
TPSF (N4) retained behind --tpsf.

Figures -> workspace/outputs/PSF/.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sigpy as sp
import asap_recon as ar

CFG = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "config.json")))
DUMP = CFG["recon_io_dyn_027JC"]
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "PSF")
os.makedirs(OUT, exist_ok=True)

NAV_N = 64
MS = 240

meta = json.load(open(os.path.join(DUMP, "meta.json")))
npts = meta["npts"]
traj = np.stack([np.load(os.path.join(DUMP, f"traj{a}.npy")) for a in "xyz"], 1)
T = traj.reshape(832, npts, 3)

n = NAV_N
c = n // 2
zz, yy, xx = np.mgrid[:n, :n, :n]
R = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2)


def traj_rad(i0, nilv):
    return ar.grid_to_radians(T[i0:i0 + nilv].reshape(-1, 3).astype(float), MS)


def dcf(tr):
    ones = np.ones(len(tr), dtype=complex)
    img = ar.adjoint(tr, ones, n)
    dens = np.abs(ar.forward(tr, img))
    w = 1.0 / np.clip(dens, dens.max() * 1e-4, None)
    return w / w.mean()


def psf(tr, w):
    delta = np.zeros((n, n, n), dtype=complex)
    delta[c, c, c] = 1.0
    y = ar.forward(tr, delta)
    img = ar.adjoint(tr, y * w, n)
    return img / img[c, c, c]          # complex normalize: mainlobe peak -> 1+0j


def fwhm_1d(prof):
    p = np.abs(prof) / np.abs(prof).max()
    pk = np.argmax(p)
    def cross(side):
        idxs = range(pk, len(p) - 1) if side > 0 else range(pk, 0, -1)
        for i in idxs:
            j = i + side
            if p[j] < 0.5 <= p[i]:
                return i + side * (p[i] - 0.5) / (p[i] - p[j])
        return pk + side * (len(p) // 2)
    return cross(+1) - cross(-1)


def field_stats(F, mask):
    """Noise-scale stats of a complex field on mask. sigma from RMS (Rayleigh)."""
    v = F[mask]
    sigma = np.sqrt(np.mean(np.abs(v) ** 2) / 2.0)
    mx = np.abs(v).max()
    nn = v.size
    exp_max = sigma * np.sqrt(2 * np.log(nn))
    comp = np.concatenate([v.real, v.imag])
    kurt = float(((comp - comp.mean()) ** 4).mean() / (comp.var() ** 2) - 3.0)
    pk = np.unravel_index(np.argmax(np.where(mask, np.abs(F), 0)), F.shape)
    return dict(sigma=float(sigma), max=float(mx), exp_max=float(exp_max),
                coh=float(mx / exp_max), kurt=kurt, peak=pk, n=nn)


def shell_profiles(a, nb=60):
    rb = np.linspace(0, c, nb)
    mean_p, max_p = [], []
    for i in range(nb - 1):
        s = (R >= rb[i]) & (R < rb[i + 1])
        mean_p.append(a[s].mean() if s.any() else np.nan)
        max_p.append(a[s].max() if s.any() else np.nan)
    return rb[:-1], np.array(mean_p), np.array(max_p)


# ------------------------------------------------------------------ reference
print("full-832 reference PSF ...")
tr_full = traj_rad(0, 832)
P_full = psf(tr_full, dcf(tr_full))
a_full = np.abs(P_full)
fw_full = np.mean([fwhm_1d(a_full[:, c, c]), fwhm_1d(a_full[c, :, c]),
                   fwhm_1d(a_full[c, c, :])])
side_full = field_stats(P_full, R > 2 * fw_full)
print(f"[full-832] FWHM={fw_full:.2f} vox  peak-side={side_full['max']:.4f} "
      f"(aperture ring)  sigma={side_full['sigma']:.5f}")

windows = [("aligned-26 (block 0)", 0, 26),
           ("old-20 (ilv 0-19)", 0, 20),
           ("aligned-26 (block 10)", 260, 26)]

fig, axes = plt.subplots(3, 3, figsize=(16.5, 13))
for row, (label, i0, nilv) in enumerate(windows):
    tr = traj_rad(i0, nilv)
    P = psf(tr, dcf(tr))
    a = np.abs(P)
    fw = np.mean([fwhm_1d(a[:, c, c]), fwhm_1d(a[c, :, c]), fwhm_1d(a[c, c, :])])
    alias = P - P_full                     # aperture ring cancels here
    aal = np.abs(alias)
    st = field_stats(alias, R > 2 * fw)    # alias field beyond mainlobe
    st_all = field_stats(alias, R >= 0)    # and over the whole FOV
    dvec = np.array(st["peak"], float) - c
    dvec /= max(np.linalg.norm(dvec), 1e-9)
    print(f"[{label}] FWHM={fw:.2f}  alias-max={st['max']:.4f} at r="
          f"{R[st['peak']]:.1f} dir ({dvec[0]:+.2f},{dvec[1]:+.2f},{dvec[2]:+.2f})  "
          f"sigma={st['sigma']:.5f}  max/exp_max={st['coh']:.2f}  kurt={st['kurt']:.2f}")

    # col 0: central slice of |alias| (log)
    im = axes[row, 0].imshow(np.log10(aal[c] + 1e-5), cmap="magma", vmin=-4, vmax=-0.5)
    axes[row, 0].set_title(f"{label}\nlog10|alias PSF| (window - full)")
    plt.colorbar(im, ax=axes[row, 0], shrink=0.8)

    # col 1: radial shell mean/max of PSF vs full reference
    rb, pm, px = shell_profiles(a)
    _, pmF, pxF = shell_profiles(a_full)
    axes[row, 1].semilogy(rb, px, color="crimson", lw=1.0, label="window shell max")
    axes[row, 1].semilogy(rb, pm, color="steelblue", lw=1.0, label="window shell mean")
    axes[row, 1].semilogy(rb, pxF, color="k", lw=1.0, label="FULL-832 shell max")
    axes[row, 1].semilogy(rb, pmF, color="gray", lw=0.8, ls="--", label="FULL-832 shell mean")
    axes[row, 1].set_title(f"|PSF| radial  FWHM {fw:.2f} vox (full {fw_full:.2f})")
    axes[row, 1].set_xlabel("radius (vox)")
    axes[row, 1].legend(fontsize=6); axes[row, 1].grid(alpha=0.3)

    # col 2: radial profile of the ALIAS field + noise-scale lines
    rbA, pmA, pxA = shell_profiles(aal)
    axes[row, 2].semilogy(rbA, pxA, color="crimson", lw=1.0, label="alias shell max")
    axes[row, 2].semilogy(rbA, pmA, color="steelblue", lw=1.0, label="alias shell mean")
    axes[row, 2].axhline(st["sigma"], color="gray", ls="--", lw=0.9,
                         label=f"sigma {st['sigma']:.4f}")
    axes[row, 2].axhline(st["exp_max"], color="green", ls=":", lw=1.0,
                         label=f"noise-like max {st['exp_max']:.4f}")
    axes[row, 2].plot([R[st["peak"]]], [st["max"]], "o", color="red", ms=6,
                      label=f"alias max {st['max']:.4f}")
    axes[row, 2].set_title(f"alias field  max/noise-like-max = {st['coh']:.2f}  "
                           f"kurt {st['kurt']:.2f}")
    axes[row, 2].set_ylim(3e-3, 0.5)
    axes[row, 2].set_xlabel("radius (vox)")
    axes[row, 2].legend(fontsize=6); axes[row, 2].grid(alpha=0.3)

fig.suptitle("Alias PSF = window PSF − full-832 PSF   (64$^3$ nav settings, DCF, v3_dyn 027JC)\n"
             "aperture/Gibbs ring cancels in the subtraction; sigma = sqrt(E|x|$^2$/2), "
             "noise-like max = sigma·sqrt(2 ln n)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(OUT, "alias_psf_battery.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
