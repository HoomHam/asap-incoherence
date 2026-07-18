// WebWorker: owns the WASM module (kasap generator + gridding NUFFT),
// the current trajectory, and the full-set PSF cache. All heavy numerics here.

import createModule from './wasm/kasap.js';
import wasmUrl from './wasm/kasap.wasm?url';
import goldenUrl from './assets/golden_rad.f32.bin?url';
import type { GenParams, TrajData, PsfResult, WorkerRequest, GoldenReport } from './types';
import { absVol, subVol, fwhm3, shellProfiles, fieldStats, dipole, } from './metrics';

interface EmModule {
  HEAPF32: Float32Array;
  HEAPF64: Float64Array;
  _kasap_generate(...a: number[]): number;
  _kasap_get_kx(): number; _kasap_get_ky(): number; _kasap_get_kz(): number;
  _kasap_get_basis(): number; _kasap_get_reprot(): number;
  _nufft_psf(wx: number, wy: number, wz: number, M: number, n: number): number;
  _nufft_get_psf_re(): number; _nufft_get_psf_im(): number;
  _wasm_malloc(n: number): number; _wasm_free(p: number): void;
}

let M: EmModule;
const ready = (createModule as (o: object) => Promise<EmModule>)({
  locateFile: (p: string) => (p.endsWith('.wasm') ? wasmUrl : p),
}).then(m => { M = m; });

// current trajectory (copies, JS side)
let traj: TrajData | null = null;
let curParams: GenParams | null = null;
const fullPsfCache = new Map<number, { re: Float32Array; im: Float32Array; fwhm: number }>();

function post(msg: object) { (self as unknown as Worker).postMessage(msg); }

function generate(params: GenParams): TrajData {
  const total = M._kasap_generate(
    params.NI, params.NPTS, params.NREPS,
    params.n, params.at, params.fov, params.ms,
    params.optimize ? 1 : 0, (params.rotAngleDeg * Math.PI) / 180,
    params.axisMode === 'fixed' ? 1 : 0, params.axis[0], params.axis[1], params.axis[2],
    params.t0, params.dt, params.maxs, params.maxg,
  );
  if (!total) throw new Error('kasap_generate failed (allocation)');
  const grab = (ptr: number, len: number) => M.HEAPF32.slice(ptr >> 2, (ptr >> 2) + len);
  const kx = grab(M._kasap_get_kx(), total);
  const ky = grab(M._kasap_get_ky(), total);
  const kz = grab(M._kasap_get_kz(), total);
  let nanCount = 0;
  for (let i = 0; i < total; i++)
    if (Number.isNaN(kx[i]) || Number.isNaN(ky[i]) || Number.isNaN(kz[i])) nanCount++;
  traj = {
    total, NI: params.NI, NPTS: params.NPTS, NREPS: params.NREPS,
    kx, ky, kz,
    basis: grab(M._kasap_get_basis(), params.NI * 3),
    reprot: grab(M._kasap_get_reprot(), params.NREPS * 3),
    kmax: params.ms / (2 * params.fov),
    fov: params.fov, ms: params.ms,
    nanCount,
  };
  curParams = params;
  fullPsfCache.clear();
  return traj;
}

/** Sample indices of the given global interleaves (g = irep*NI + j). */
function sampleIndices(ilvs: number[]): Int32Array {
  const t = traj!;
  const idx = new Int32Array(ilvs.length * t.NPTS);
  let o = 0;
  for (const g of ilvs) {
    const irep = Math.floor(g / t.NI), j = g % t.NI;
    const base = irep * t.NI * t.NPTS + j * t.NPTS;
    for (let p = 0; p < t.NPTS; p++) idx[o++] = base + p;
  }
  return idx;
}

/** Run the WASM PSF pipeline on the given sample set. Returns copies. */
function runPsf(idx: Int32Array, n: number): { re: Float32Array; im: Float32Array } {
  const t = traj!;
  const conv = (2 * Math.PI * t.fov) / t.ms; // omega = 2*pi*k_phys*fov/ms (F14)
  // drop limit-violation (NaN) samples — they would poison the whole grid
  const valid: number[] = [];
  for (let i = 0; i < idx.length; i++) {
    const l = idx[i];
    if (!Number.isNaN(t.kx[l]) && !Number.isNaN(t.ky[l]) && !Number.isNaN(t.kz[l])) valid.push(l);
  }
  const Msamp = valid.length;
  if (!Msamp) throw new Error('all samples of this ensemble are limit-violations (NaN) — adjust n / at / slew');
  const px = M._wasm_malloc(Msamp * 8), py = M._wasm_malloc(Msamp * 8), pz = M._wasm_malloc(Msamp * 8);
  try {
    // views must be taken AFTER all allocations (memory growth invalidates them)
    const wx = M.HEAPF64.subarray(px >> 3, (px >> 3) + Msamp);
    const wy = M.HEAPF64.subarray(py >> 3, (py >> 3) + Msamp);
    const wz = M.HEAPF64.subarray(pz >> 3, (pz >> 3) + Msamp);
    for (let i = 0; i < Msamp; i++) {
      const l = valid[i];
      wx[i] = t.kx[l] * conv; wy[i] = t.ky[l] * conv; wz[i] = t.kz[l] * conv;
    }
    if (!M._nufft_psf(px, py, pz, Msamp, n)) throw new Error(`nufft_psf failed (n=${n}, M=${Msamp})`);
    const Nv = n * n * n;
    return {
      re: M.HEAPF32.slice(M._nufft_get_psf_re() >> 2, (M._nufft_get_psf_re() >> 2) + Nv),
      im: M.HEAPF32.slice(M._nufft_get_psf_im() >> 2, (M._nufft_get_psf_im() >> 2) + Nv),
    };
  } finally {
    M._wasm_free(px); M._wasm_free(py); M._wasm_free(pz);
  }
}

function fullPsf(n: number, id: number): { re: Float32Array; im: Float32Array; fwhm: number } {
  const hit = fullPsfCache.get(n);
  if (hit) return hit;
  const t = traj!;
  post({ id, op: 'progress', msg: `computing full-set reference PSF (${t.total.toLocaleString()} samples, n=${n}) — cached per parameter set…` });
  const all = new Int32Array(t.total);
  for (let i = 0; i < t.total; i++) all[i] = i;
  const p = runPsf(all, n);
  const entry = { ...p, fwhm: fwhm3(p.re, p.im, n) };
  fullPsfCache.set(n, entry);
  return entry;
}

function computePsf(ilvs: number[], n: number, label: string, id: number): PsfResult {
  if (!traj) throw new Error('generate a trajectory first');
  const full = fullPsf(n, id);
  post({ id, op: 'progress', msg: `computing ensemble PSF (${ilvs.length} interleaves)…` });
  const idx = sampleIndices(ilvs);
  const p = runPsf(idx, n);
  const fwhm = fwhm3(p.re, p.im, n);
  const [aRe, aIm] = subVol(p.re, p.im, full.re, full.im);
  const magP = absVol(p.re, p.im);
  const magF = absVol(full.re, full.im);
  const magA = absVol(aRe, aIm);
  return {
    n, nIlv: ilvs.length, nSamples: idx.length,
    psfRe: p.re, psfIm: p.im,
    fwhm, fwhmFull: full.fwhm,
    shell: shellProfiles(magP, n),
    shellFull: shellProfiles(magF, n),
    aliasShell: shellProfiles(magA, n),
    aliasStats: fieldStats(aRe, aIm, n, 2 * fwhm),
    dipole: dipole(traj.kx, traj.ky, traj.kz, idx, traj.kmax),
    aliasRe: aRe, aliasIm: aIm,
    fullRe: full.re, fullIm: full.im,
    label,
  };
}

async function golden(id: number): Promise<GoldenReport> {
  post({ id, op: 'progress', msg: 'fetching golden v3 trajectory (5 MB)…' });
  const resp = await fetch(goldenUrl);
  if (!resp.ok) throw new Error('golden trajectory fetch failed');
  const buf = new Float32Array(await resp.arrayBuffer());
  const Mtot = buf.length / 3;
  const n = 64;
  const alloc = (cnt: number) => M._wasm_malloc(cnt * 8);
  const px = alloc(Mtot), py = alloc(Mtot), pz = alloc(Mtot);
  try {
    const wx = M.HEAPF64.subarray(px >> 3, (px >> 3) + Mtot);
    const wy = M.HEAPF64.subarray(py >> 3, (py >> 3) + Mtot);
    const wz = M.HEAPF64.subarray(pz >> 3, (pz >> 3) + Mtot);
    wx.set(buf.subarray(0, Mtot)); wy.set(buf.subarray(Mtot, 2 * Mtot)); wz.set(buf.subarray(2 * Mtot));
    post({ id, op: 'progress', msg: `golden full-set PSF (${Mtot.toLocaleString()} samples)…` });
    if (!M._nufft_psf(px, py, pz, Mtot, n)) throw new Error('golden full psf failed');
    const Nv = n * n * n;
    const fRe = M.HEAPF32.slice(M._nufft_get_psf_re() >> 2, (M._nufft_get_psf_re() >> 2) + Nv);
    const fIm = M.HEAPF32.slice(M._nufft_get_psf_im() >> 2, (M._nufft_get_psf_im() >> 2) + Nv);
    post({ id, op: 'progress', msg: 'golden block-0 PSF…' });
    const Mb = 26 * (Mtot / 832);
    if (!M._nufft_psf(px, py, pz, Mb, n)) throw new Error('golden b0 psf failed');
    const bRe = M.HEAPF32.slice(M._nufft_get_psf_re() >> 2, (M._nufft_get_psf_re() >> 2) + Nv);
    const bIm = M.HEAPF32.slice(M._nufft_get_psf_im() >> 2, (M._nufft_get_psf_im() >> 2) + Nv);

    const fullFwhm = fwhm3(fRe, fIm, n);
    const magF = absVol(fRe, fIm);
    const c = n >> 1;
    let sideMax = 0;
    for (let x = 0; x < n; x++)
      for (let y = 0; y < n; y++)
        for (let z = 0; z < n; z++) {
          const r = Math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2);
          if (r > 2 * fullFwhm) sideMax = Math.max(sideMax, magF[(x * n + y) * n + z]);
        }
    const b0Fwhm = fwhm3(bRe, bIm, n);
    const [aRe, aIm] = subVol(bRe, bIm, fRe, fIm);
    const st = fieldStats(aRe, aIm, n, 2 * b0Fwhm);
    return {
      fullFwhm, fullSideMax: sideMax,
      b0Fwhm, b0AliasMax: st.max, b0Sigma: st.sigma, b0Coh: st.coh,
    };
  } finally {
    M._wasm_free(px); M._wasm_free(py); M._wasm_free(pz);
  }
}

self.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  try {
    await ready;
    if (req.op === 'generate') {
      const t = generate(req.params);
      // transfer copies (keep worker-side originals)
      post({
        id: req.id, op: 'generate',
        traj: {
          ...t,
          kx: t.kx.slice(), ky: t.ky.slice(), kz: t.kz.slice(),
          basis: t.basis.slice(), reprot: t.reprot.slice(),
        },
      });
    } else if (req.op === 'psf') {
      const result = computePsf(req.ilvs, req.n, req.label, req.id);
      post({ id: req.id, op: 'psf', result });
    } else if (req.op === 'golden') {
      const report = await golden(req.id);
      post({ id: req.id, op: 'golden', report });
    }
  } catch (e) {
    post({ id: req.id, op: 'error', msg: e instanceof Error ? e.message : String(e) });
  }
};
