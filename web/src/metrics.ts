// Numerics ported from reference_py/_psf_incoherence.py and _psf_fan_view.py.
// Volume layout: v[(x*n + y)*n + z], center c = n/2 (matches the C module).

import type { ShellProfile, FieldStats, TrajData } from './types';

export interface DensityProfile {
  r: Float64Array;     // shell centers, grid units (1 unit = 1/FOV)
  dens: Float64Array;  // samples per (1/FOV)^3 k-space cell
  rN: number;          // local Nyquist radius: shells inside have dens >= 1
  kmaxGrid: number;    // ms/2
}

/** Shell-histogram sample density of an ensemble, in grid units.
    Nyquist-complete where >= 1 sample per unit-volume cell; rN is the first
    radius (past the r<2 center pileup) where density falls below that. */
export function shellDensity(t: TrajData, ilvs: number[]): DensityProfile {
  const kmaxGrid = t.ms / 2;
  const nb = Math.ceil(kmaxGrid) + 2;
  const count = new Float64Array(nb);
  for (const g of ilvs) {
    const base = g * t.NPTS;  // lin = pt + (irep*NI + j)*NPTS
    for (let p = 0; p < t.NPTS; p++) {
      const x = t.kx[base + p], y = t.ky[base + p], z = t.kz[base + p];
      if (Number.isNaN(x)) continue;
      const b = Math.floor(Math.hypot(x, y, z) * t.fov);
      if (b < nb) count[b]++;
    }
  }
  const r = new Float64Array(nb), dens = new Float64Array(nb);
  for (let b = 0; b < nb; b++) {
    const rc = b + 0.5;
    r[b] = rc;
    dens[b] = count[b] / (4 * Math.PI * rc * rc);
  }
  let rN = kmaxGrid;
  for (let b = 2; b < nb; b++) if (dens[b] < 1) { rN = b; break; }
  return { r, dens, rN, kmaxGrid };
}

export function absVol(re: Float32Array, im: Float32Array): Float32Array {
  const out = new Float32Array(re.length);
  for (let i = 0; i < re.length; i++) out[i] = Math.hypot(re[i], im[i]);
  return out;
}

export function subVol(aRe: Float32Array, aIm: Float32Array, bRe: Float32Array, bIm: Float32Array): [Float32Array, Float32Array] {
  const re = new Float32Array(aRe.length), im = new Float32Array(aRe.length);
  for (let i = 0; i < aRe.length; i++) { re[i] = aRe[i] - bRe[i]; im[i] = aIm[i] - bIm[i]; }
  return [re, im];
}

/** FWHM of |profile| by linear interpolation of half-max crossings. */
export function fwhm1d(prof: Float64Array): number {
  const p = new Float64Array(prof.length);
  let mx = 0;
  for (const v of prof) mx = Math.max(mx, Math.abs(v));
  for (let i = 0; i < prof.length; i++) p[i] = Math.abs(prof[i]) / mx;
  let pk = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > p[pk]) pk = i;
  const cross = (side: number): number => {
    for (let i = pk; side > 0 ? i < p.length - 1 : i > 0; i += side) {
      const j = i + side;
      if (p[j] < 0.5 && 0.5 <= p[i]) return i + side * (p[i] - 0.5) / (p[i] - p[j]);
    }
    return pk + side * (p.length >> 1);
  };
  return cross(1) - cross(-1);
}

/** 3-axis mean FWHM of |P| through the center. */
export function fwhm3(re: Float32Array, im: Float32Array, n: number): number {
  const c = n >> 1;
  const line = (get: (i: number) => number) => {
    const p = new Float64Array(n);
    for (let i = 0; i < n; i++) p[i] = get(i);
    return fwhm1d(p);
  };
  const a = (x: number, y: number, z: number) => {
    const l = (x * n + y) * n + z;
    return Math.hypot(re[l], im[l]);
  };
  return (line(i => a(i, c, c)) + line(i => a(c, i, c)) + line(i => a(c, c, i))) / 3;
}

/** Radial shell mean/max of a magnitude volume, 60 bins to n/2. */
export function shellProfiles(mag: Float32Array, n: number, nb = 60): ShellProfile {
  const c = n >> 1;
  const edges = new Float64Array(nb);
  for (let i = 0; i < nb; i++) edges[i] = (c * i) / (nb - 1);
  const sum = new Float64Array(nb - 1), cnt = new Float64Array(nb - 1), mx = new Float64Array(nb - 1).fill(NaN);
  for (let x = 0; x < n; x++)
    for (let y = 0; y < n; y++)
      for (let z = 0; z < n; z++) {
        const r = Math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2);
        // bin index: edges are linspace(0, c, nb)
        const b = Math.floor(r / (c / (nb - 1)));
        if (b < 0 || b >= nb - 1) continue;
        const v = mag[(x * n + y) * n + z];
        sum[b] += v; cnt[b]++;
        if (!(v <= mx[b])) mx[b] = v;
      }
  const mean = new Float64Array(nb - 1);
  for (let i = 0; i < nb - 1; i++) mean[i] = cnt[i] ? sum[i] / cnt[i] : NaN;
  return { r: edges.slice(0, nb - 1), mean, max: mx };
}

/** Rayleigh-correct stats of a complex field over the mask r > rMin.
 *  sigma = sqrt(E|x|^2 / 2); noise-like max = sigma*sqrt(2 ln n). (F6) */
export function fieldStats(re: Float32Array, im: Float32Array, n: number, rMin: number): FieldStats {
  const c = n >> 1;
  let s2 = 0, count = 0, mx = 0;
  let mean = 0, m2sum = 0; // for component moments
  let pk: [number, number, number] = [0, 0, 0];
  // pass 1: sigma, max, peak
  for (let x = 0; x < n; x++)
    for (let y = 0; y < n; y++)
      for (let z = 0; z < n; z++) {
        const r = Math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2);
        if (r <= rMin) continue;
        const l = (x * n + y) * n + z;
        const a2 = re[l] * re[l] + im[l] * im[l];
        s2 += a2; count++;
        const a = Math.sqrt(a2);
        if (a > mx) { mx = a; pk = [x, y, z]; }
        mean += re[l] + im[l];
      }
  const sigma = Math.sqrt(s2 / count / 2);
  mean /= 2 * count;
  // pass 2: kurtosis of pooled re/im components
  let v2 = 0, v4 = 0;
  for (let x = 0; x < n; x++)
    for (let y = 0; y < n; y++)
      for (let z = 0; z < n; z++) {
        const r = Math.sqrt((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2);
        if (r <= rMin) continue;
        const l = (x * n + y) * n + z;
        for (const comp of [re[l], im[l]]) {
          const d = comp - mean;
          v2 += d * d; v4 += d * d * d * d;
        }
      }
  v2 /= 2 * count; v4 /= 2 * count;
  const kurt = v4 / (v2 * v2) - 3;
  const expMax = sigma * Math.sqrt(2 * Math.log(count));
  const dv: [number, number, number] = [pk[0] - c, pk[1] - c, pk[2] - c];
  const dn = Math.max(Math.hypot(...dv), 1e-9);
  return {
    sigma, max: mx, expMax, coh: mx / expMax, kurt,
    peakR: Math.hypot(...dv),
    peakDir: [dv[0] / dn, dv[1] / dn, dv[2] / dn],
  };
}

/** Balance metric: |mean unit k-direction| over samples with |k| > 0.2 kmax. */
export function dipole(kx: Float32Array, ky: Float32Array, kz: Float32Array,
                       idx: Int32Array, kmax: number): number {
  let sx = 0, sy = 0, sz = 0, cnt = 0;
  const thr = 0.2 * kmax;
  for (let i = 0; i < idx.length; i++) {
    const l = idx[i];
    const r = Math.hypot(kx[l], ky[l], kz[l]);
    if (r <= thr) continue;
    sx += kx[l] / r; sy += ky[l] / r; sz += kz[l] / r; cnt++;
  }
  if (!cnt) return 0;
  return Math.hypot(sx / cnt, sy / cnt, sz / cnt);
}

/** Trilinear sample of volume at fractional (x,y,z). Nearest at edges. */
export function trilinear(vol: Float32Array, n: number, x: number, y: number, z: number): number {
  const cl = (v: number) => Math.min(Math.max(v, 0), n - 1);
  x = cl(x); y = cl(y); z = cl(z);
  const x0 = Math.floor(x), y0 = Math.floor(y), z0 = Math.floor(z);
  const x1 = Math.min(x0 + 1, n - 1), y1 = Math.min(y0 + 1, n - 1), z1 = Math.min(z0 + 1, n - 1);
  const fx = x - x0, fy = y - y0, fz = z - z0;
  const v = (xi: number, yi: number, zi: number) => vol[(xi * n + yi) * n + zi];
  return (
    v(x0, y0, z0) * (1 - fx) * (1 - fy) * (1 - fz) +
    v(x1, y0, z0) * fx * (1 - fy) * (1 - fz) +
    v(x0, y1, z0) * (1 - fx) * fy * (1 - fz) +
    v(x0, y0, z1) * (1 - fx) * (1 - fy) * fz +
    v(x1, y1, z0) * fx * fy * (1 - fz) +
    v(x1, y0, z1) * fx * (1 - fy) * fz +
    v(x0, y1, z1) * (1 - fx) * fy * fz +
    v(x1, y1, z1) * fx * fy * fz
  );
}

/** Fan view |vol|(r, theta) in a central plane. plane: 'xy' | 'xz' | 'yz'.
 *  Returns row-major [theta][r], theta 0..359 deg step 1, r 0..n/2-2 step 0.25.
 *  Axis convention matches reference: vol index order (x,y,z); reference volume
 *  order (z,y,x) with planes xy=(ax 2, ax 1), xz=(2,0), yz=(1,0) — mapped here
 *  to (x,y), (x,z)->... plane 'xy': vary idx0,idx1; 'xz': idx0,idx2; 'yz': idx1,idx2. */
export function fanView(mag: Float32Array, n: number, plane: 'xy' | 'xz' | 'yz'):
    { rs: Float64Array; fan: Float32Array; nTheta: number } {
  const c = n >> 1;
  const rMax = c - 2;
  const nr = Math.floor(rMax / 0.25);
  const rs = new Float64Array(nr);
  for (let i = 0; i < nr; i++) rs[i] = i * 0.25;
  const nTheta = 360;
  const fan = new Float32Array(nTheta * nr);
  for (let t = 0; t < nTheta; t++) {
    const th = (t * Math.PI) / 180;
    const ca = Math.cos(th), sa = Math.sin(th);
    for (let i = 0; i < nr; i++) {
      const a = c + rs[i] * ca, b = c + rs[i] * sa;
      let x = c, y = c, z = c;
      if (plane === 'xy') { x = a; y = b; }
      else if (plane === 'xz') { x = a; z = b; }
      else { y = a; z = b; }
      fan[t * nr + i] = trilinear(mag, n, x, y, z);
    }
  }
  return { rs, fan, nTheta };
}
