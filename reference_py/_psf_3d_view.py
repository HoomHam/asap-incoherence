"""3D visualization of the full-832 PSF (aperture rings) and the block-0 alias
field (scratch tier). Interactive plotly HTML (drag to rotate) + static PNG.

Cutaway: half the volume (y > center) removed so the ring shells around the
mainlobe are visible from inside.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
import plotly.graph_objects as go

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
a_full = np.abs(P_full)
a_alias = np.abs(P_win - P_full)

specs = [
    ("full-832 |PSF| — mainlobe + aperture ring shells", a_full,
     [(0.5, "gold", 1.0), (0.03, "crimson", 0.35), (0.008, "steelblue", 0.15)],
     "psf3d_full"),
    ("block-0 alias |PSF_26 - PSF_832| — alias speckle field", a_alias,
     [(0.10, "gold", 1.0), (0.06, "crimson", 0.4), (0.035, "steelblue", 0.12)],
     "psf3d_alias"),
]

for title, vol, levels, stem in specs:
    # ---------------- interactive plotly (cutaway y > c) ----------------
    fig = go.Figure()
    for lev, color, op in levels:
        try:
            verts, faces, _, _ = marching_cubes(vol, lev)
        except ValueError:
            continue
        keep = np.ones(len(faces), bool)
        vy = verts[:, 1]
        keep = ~(vy[faces].min(axis=1) > c)          # drop faces fully in y>c half
        f = faces[keep]
        fig.add_trace(go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=color, opacity=op, name=f"|PSF|={lev}", showlegend=True))
    fig.update_layout(
        title=f"{title}<br><sup>cutaway: front half removed; gold=mainlobe/peak, "
              f"red/blue = lower levels</sup>",
        scene=dict(xaxis_title="x (vox)", yaxis_title="y (vox)", zaxis_title="z (vox)",
                   aspectmode="cube",
                   camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9))),
        width=950, height=850)
    html = os.path.join(OUT, f"{stem}.html")
    fig.write_html(html, include_plotlyjs=True)
    print("saved", html)

# ---------------- static matplotlib PNG, nice angle ----------------
fig = plt.figure(figsize=(16, 8))
for k, (title, vol, levels, _) in enumerate(specs):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    for lev, color, op in levels:
        try:
            verts, faces, _, _ = marching_cubes(vol, lev)
        except ValueError:
            continue
        vy = verts[:, 1]
        keep = ~(vy[faces].min(axis=1) > c)
        mesh = Poly3DCollection(verts[faces[keep]], alpha=op)
        mesh.set_facecolor(color); mesh.set_edgecolor("none")
        ax.add_collection3d(mesh)
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_zlim(0, n)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-55)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("z"); ax.set_ylabel("y"); ax.set_zlabel("x")
fig.suptitle("3D PSF structure, cutaway view (front half removed)", fontsize=13)
out = os.path.join(OUT, "psf3d_static.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
