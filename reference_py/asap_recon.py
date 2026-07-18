"""ASAP spiral recon baseline — FINUFFT operator, own thin pipeline.

Design (handoff D1/D3): library NUFFT as A/A^H, bins enter only as
per-sample weights. recon(traj, data, sample_weights=None) is the whole API.

Units contract
--------------
Steve's saved trajectories (trajx/y/z.npy, dumped by results.dyn_recon) are in
GRID-INDEX units: g = k * MS/IS + MS/2, with k in delta-k (cycles/FOV).
FINUFFT wants nonuniform coordinates in radians. Conversion:

    x_rad = (g - MS/2) * 2*pi / MS  =  k * 2*pi / IS

so an image grid of n=IS points per dim spans exactly one FOV.

Sign convention: Steve/Faraz apply a *forward* FFT k->image, i.e. their image
is the isign=-1 adjoint (a conjugated/flipped twin of the textbook isign=+1
recon). Default here is isign=-1 to match Steve voxel-for-voxel; pass
isign=+1 for the textbook orientation.
"""

import numpy as np
import finufft

MS_DEFAULT = 240
IS_DEFAULT = 100


# ---------------------------------------------------------------- data loading

def load_steve_npy(folder):
    """Load the arrays results.dyn_recon dumps (results.py:258-262).

    Returns dict with:
      traj : (Nuniq, 3) float64, grid-index units
      acq  : (M,) complex64, channel-0 samples, spike/exclusion zeros in place
      bins : (Nilv,) float64 bin centers per interleave (or None)
    """
    import os
    p = lambda name: os.path.join(folder, name)
    traj = np.stack([np.load(p(f"traj{ax}.npy")) for ax in "xyz"], axis=1)
    acq = np.load(p("acq.npy"))
    bins = np.load(p("bins.npy")) if os.path.exists(p("bins.npy")) else None
    return {"traj": traj, "acq": acq, "bins": bins}


def grid_to_radians(traj_grid, MS=MS_DEFAULT):
    """Grid-index trajectory -> radians for FINUFFT. Wraps nothing; FINUFFT
    accepts any real coords, but values should land in [-pi, pi) when the
    trajectory respects kmax <= IS/2."""
    return (traj_grid - MS / 2) * (2 * np.pi / MS)


def tile_traj(traj, n_samples):
    """Steve grids all acquired interleaves against the unique trajectory,
    index wrapping kidx = idx %% nuniq (recon.py:103). Replicate that."""
    nuniq = traj.shape[0]
    reps = int(np.ceil(n_samples / nuniq))
    return np.tile(traj, (reps, 1))[:n_samples]


# ---------------------------------------------------------------- operators

def adjoint(traj_rad, data, n, weights=None, isign=-1, eps=1e-9):
    """A^H: nonuniform samples -> n^3 image. weights = DCF and/or bin/soft
    weights folded into the data (sample_weights of the handoff API)."""
    c = np.ascontiguousarray(data, dtype=np.complex128)
    if weights is not None:
        c = c * weights
    x, y, z = (np.ascontiguousarray(traj_rad[:, i]) for i in range(3))
    return finufft.nufft3d1(x, y, z, c, (n, n, n), isign=isign, eps=eps)


def forward(traj_rad, img, isign=None, eps=1e-9):
    """A: n^3 image -> nonuniform samples. isign defaults to the adjoint's
    conjugate (so <Ax, y> == <x, A^H y> holds for the pair used here)."""
    if isign is None:
        isign = +1  # conjugate of adjoint default isign=-1
    x, y, z = (np.ascontiguousarray(traj_rad[:, i]) for i in range(3))
    return finufft.nufft3d2(x, y, z, np.ascontiguousarray(img, dtype=np.complex128),
                            isign=isign, eps=eps)


# ---------------------------------------------------------------- DCF
# Deleted 2026-06-11: the PSF-as-kernel Pipe-Menon variant underperformed
# badly on real data (corr 0.38 vs Steve, worse than the plain adjoint) and
# CG makes one-shot DCF redundant. If a fast preview-with-DCF is ever needed,
# implement proper compact-kernel Pipe-Menon (grid/degrid with a KB kernel,
# not the full PSF) — see Faraz's iterative_dcf_fa_20190910.m for reference.


# ---------------------------------------------------------------- CG inverse

def cg_recon(traj_rad, data, n, sample_weights=None, n_iter=20, lam=0.0,
             eps=1e-7, verbose=True):
    """Solve (A^H W A + lam I) x = A^H W s by conjugate gradients.
    sample_weights: per-sample weights W (bin weights, reliability, DCF as
    preconditioner-ish). None -> unweighted least squares."""
    w = np.ones(traj_rad.shape[0]) if sample_weights is None else sample_weights

    def normal_op(x_img):
        return adjoint(traj_rad, w * forward(traj_rad, x_img, eps=eps), n,
                       eps=eps) + lam * x_img

    b = adjoint(traj_rad, w * data, n, eps=eps)
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = np.vdot(r, r).real
    for it in range(n_iter):
        Ap = normal_op(p)
        alpha = rs / np.vdot(p, Ap).real
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.vdot(r, r).real
        if verbose:
            print(f"  cg iter {it+1:3d}  residual {np.sqrt(rs_new):.4e}")
        if np.sqrt(rs_new) < 1e-12:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


# ---------------------------------------------------------------- handoff API

def recon(traj, data, sample_weights=None, method="cg",
          MS=MS_DEFAULT, IS=IS_DEFAULT, cg_iters=20, lam=0.0):
    """The API of handoff decision D3.

    traj : (Nuniq,3) grid-index units (Steve convention) — auto-tiled to data
    data : (M,) complex samples (zeros = excluded, contribute nothing)
    sample_weights : (M,) or None. Static = None. Binned recon = call once
        per bin with that bin's weight vector. Soft bins, spike masks,
        reliability — all just weights.
    method : 'cg' (default, the method of record) | 'adjoint' (fast preview;
        density-biased — expect center-heavy blur)

    Defaults settled by the 2026-06-11 sweep (cg_tune.py): cg_iters=20,
    lam=0. Tikhonov lam is a no-op on fully-sampled data (A^H A is
    well-conditioned); noise parity with Steve's filtered gridder comes from
    a smoothing regularizer (the CS layer), not from lam.
    Returns IS^3 complex image.
    """
    traj_rad = grid_to_radians(tile_traj(np.asarray(traj, float), len(data)), MS)
    if method == "adjoint":
        return adjoint(traj_rad, data, IS, weights=sample_weights)
    if method == "cg":
        return cg_recon(traj_rad, data, IS, sample_weights=sample_weights,
                        n_iter=cg_iters, lam=lam)
    raise ValueError(f"unknown method {method!r}")
