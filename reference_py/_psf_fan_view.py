"""Un-collapsed radial view: |PSF|(r, theta) fan heatmaps (scratch tier).

Rows: full-832 PSF | window-26 block0 PSF | alias (window - full).
Cols: rays swept 360 deg in the xy / xz / yz central planes.
Rings (isotropic) -> vertical stripes. Directional alias -> horizontal streaks.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

import asap_recon as ar

CFG = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "config.json")))
DUMP = CFG["recon_io_dyn_027JC"]
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "PSF")

n, c, MS = 64, 32, 240
meta = json.load(open(os.path.join(DUMP, "meta.json")))
traj = np.stack([np.load(os.path.join(DUMP, f"traj{a}.npy")) for a in "xyz"], 1)
T = traj.reshape(832, meta["npts"], 3)


def psf_of(tg):
    tr = ar.grid_to_radians(tg.astype(float), MS)
    ones = np.ones(len(tr), dtype=complex)
    img = ar.adjoint(tr, ones, n)
    dens = np.abs(ar.forward(tr, img))
    w = 1.0 / np.clip(dens, dens.max() * 1e-4, None); w /= w.mean()
    delta = np.zeros((n, n, n), dtype=complex); delta[c, c, c] = 1.0
    P = ar.adjoint(tr, ar.forward(tr, delta) * w, n)
    return P / P[c, c, c]


print("computing PSFs ...")
P_full = psf_of(T.reshape(-1, 3))
P_win = psf_of(T[:26].reshape(-1, 3))
fields = [("full-832 |PSF|", np.abs(P_full)),
          ("window-26 (block 0) |PSF|", np.abs(P_win)),
          ("alias |PSF$_{26}$ - PSF$_{832}$|", np.abs(P_win - P_full))]

rs = np.arange(0, 30, 0.25)
thetas = np.deg2rad(np.arange(0, 360, 1.0))
# plane -> (axis a, axis b) in (z,y,x) index order of the volume
planes = [("xy", 2, 1), ("xz", 2, 0), ("yz", 1, 0)]

fig, axes = plt.subplots(3, 3, figsize=(17, 12), sharex=True, sharey=True)
for row, (label, vol) in enumerate(fields):
    for col, (pname, ax_a, ax_b) in enumerate(planes):
        fan = np.zeros((len(thetas), len(rs)))
        for k, th in enumerate(thetas):
            coords = np.full((3, len(rs)), float(c))
            coords[ax_a] = c + rs * np.cos(th)
            coords[ax_b] = c + rs * np.sin(th)
            fan[k] = map_coordinates(vol, coords, order=1, mode="nearest")
        im = axes[row, col].imshow(
            np.log10(fan + 1e-5), aspect="auto", origin="lower", cmap="magma",
            extent=[rs[0], rs[-1], 0, 360], vmin=-3.5, vmax=-0.5)
        if row == 0:
            axes[row, col].set_title(f"{pname} plane", fontsize=11)
        if col == 0:
            axes[row, col].set_ylabel(f"{label}\nray angle (deg)", fontsize=9)
        if row == 2:
            axes[row, col].set_xlabel("radius (vox)")
        plt.colorbar(im, ax=axes[row, col], shrink=0.85, label="log10|PSF|")
fig.suptitle("PSF fan view — every direction, 360°: rings = vertical stripes, "
             "directional alias = horizontal streaks  (64³ nav settings, 027JC)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(OUT, "psf_fan_view.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
