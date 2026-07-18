"""Per-interleave PSF fan views (scratch tier): 26 pages, one per spoke of
block 0. Each page: |PSF|(r,theta) in xy / xz / yz central planes, same DCF
recipe and color scale as the window fan view, so pages are comparable.
Output: workspace/outputs/PSF/fan_views_26ilv.pdf
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


rs = np.arange(0, 30, 0.25)
thetas = np.deg2rad(np.arange(0, 360, 1.0))
planes = [("xy", 2, 1), ("xz", 2, 0), ("yz", 1, 0)]


def fan(vol, ax_a, ax_b):
    F = np.zeros((len(thetas), len(rs)))
    for k, th in enumerate(thetas):
        coords = np.full((3, len(rs)), float(c))
        coords[ax_a] = c + rs * np.cos(th)
        coords[ax_b] = c + rs * np.sin(th)
        F[k] = map_coordinates(vol, coords, order=1, mode="nearest")
    return F


pdf_path = os.path.join(OUT, "fan_views_26ilv.pdf")
with PdfPages(pdf_path) as pdf:
    for i in range(26):
        vol = np.abs(psf_of(T[i]))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        for col, (pname, ax_a, ax_b) in enumerate(planes):
            F = fan(vol, ax_a, ax_b)
            im = axes[col].imshow(np.log10(F + 1e-5), aspect="auto", origin="lower",
                                  cmap="magma", extent=[rs[0], rs[-1], 0, 360],
                                  vmin=-3.5, vmax=-0.5)
            axes[col].set_title(f"{pname} plane")
            axes[col].set_xlabel("radius (vox)")
            plt.colorbar(im, ax=axes[col], shrink=0.9)
        axes[0].set_ylabel("ray angle (deg)")
        fig.suptitle(f"interleave {i} of block 0 — single-spoke |PSF| fan "
                     f"(510 samples, 64³ nav settings, DCF)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig, dpi=120)
        plt.close(fig)
        if i % 5 == 0:
            print(f"  ilv {i}/26")
print("saved", pdf_path)
