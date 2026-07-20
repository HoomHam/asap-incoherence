// Shared main<->worker message types.

export interface GenParams {
  NI: number;
  NPTS: number;
  NREPS: number;
  n: number;          // radial shape power
  at: number;         // readout duration (s)
  fov: number;        // m
  ms: number;         // matrix size
  optimize: boolean;  // Thomson optimization
  rotAngleDeg: number; // per-rep rotation angle, degrees (kasap: 90)
  axisMode: 'hedgehog' | 'fixed';
  axis: [number, number, number];
  t0: number;
  dt: number;
  maxs: number;       // T/m/s
  maxg: number;       // T/m
}

export const V3_DEFAULTS: GenParams = {
  NI: 26, NPTS: 512, NREPS: 32, n: 2.0, at: 512e-5, fov: 0.35, ms: 160,
  optimize: true, rotAngleDeg: 90, axisMode: 'hedgehog', axis: [0, 0, 1],
  t0: 4e-5, dt: 1e-5, maxs: 150, maxg: 0.04,
};

export interface TrajData {
  total: number;
  NI: number;
  NPTS: number;
  NREPS: number;
  kx: Float32Array;   // physical cycles/m, lin = pt + ilv*NPTS + irep*NI*NPTS
  ky: Float32Array;
  kz: Float32Array;
  basis: Float32Array;   // NI x 3
  reprot: Float32Array;  // NREPS x 3
  kmax: number;          // design kmax = ms/(2 fov)
  fov: number;
  ms: number;
  nanCount: number;      // samples where the generator hit a limit violation
}

export interface ShellProfile {
  r: Float64Array;
  mean: Float64Array;
  max: Float64Array;
}

export interface FieldStats {
  sigma: number;
  max: number;
  expMax: number;
  coh: number;       // max / expMax — headline incoherence number
  kurt: number;      // excess kurtosis of pooled re/im
  peakR: number;
  peakDir: [number, number, number];
}

export interface PsfResult {
  n: number;
  nIlv: number;
  nSamples: number;
  psfRe: Float32Array;   // n^3, |P| center-normalized (complex)
  psfIm: Float32Array;
  fwhm: number;
  fwhmFull: number;
  shell: ShellProfile;       // of |PSF|
  shellFull: ShellProfile;   // of |PSF_full| (same params)
  aliasShell: ShellProfile;  // of |PSF - PSF_full|
  aliasStats: FieldStats;    // over r > 2 FWHM
  dipole: number;            // |mean unit k-dir|, |k| > 0.2 kmax
  aliasRe: Float32Array;     // alias volume for fan views
  aliasIm: Float32Array;
  fullRe: Float32Array;      // full-set PSF volume
  fullIm: Float32Array;
  label: string;
  // local Nyquist radii (grid units), attached on the main thread after compute
  rN?: number;
  rNFull?: number;
  kmaxGrid?: number;
}

export type WorkerRequest =
  | { id: number; op: 'generate'; params: GenParams }
  | { id: number; op: 'psf'; ilvs: number[]; n: number; label: string }
  | { id: number; op: 'golden' };

export type WorkerResponse =
  | { id: number; op: 'generate'; traj: TrajData }
  | { id: number; op: 'psf'; result: PsfResult }
  | { id: number; op: 'golden'; report: GoldenReport }
  | { id: number; op: 'progress'; msg: string }
  | { id: number; op: 'error'; msg: string };

export interface GoldenReport {
  fullFwhm: number;
  fullSideMax: number;
  b0Fwhm: number;
  b0AliasMax: number;
  b0Sigma: number;
  b0Coh: number;
}
