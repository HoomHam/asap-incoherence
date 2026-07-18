// Plotly rendering helpers. Scientific color conventions are fixed by the
// reference figures (spec §4-§6): turbo by interleave index with a colorbar,
// magma log heatmaps with fixed scales for fan views.

import Plotly from 'plotly.js-dist-min';

type PlotlyData = Record<string, unknown>;
type PlotlyLayout = Record<string, unknown>;
import type { TrajData, PsfResult } from './types';
import { absVol, fanView } from './metrics';

const DARK = {
  paper_bgcolor: '#14161c',
  plot_bgcolor: '#14161c',
  font: { color: '#c9ccd4', size: 12 },
};
const AXIS = { gridcolor: '#2a2e3a', zerolinecolor: '#3a3f4e' };

// explicit colorscales (plotly.js lacks built-in Magma/Turbo names)
const MAGMA_HEX = ['#000004', '#1c1044', '#4f127b', '#812581', '#b5367a',
                   '#e55064', '#fb8761', '#fec287', '#fcfdbf'];
const MAGMA: [number, string][] = MAGMA_HEX.map((c, i) => [i / (MAGMA_HEX.length - 1), c]);
const TURBO_SCALE: [number, string][] = [
  [0, 'rgb(48,18,59)'], [0.125, 'rgb(70,107,227)'], [0.25, 'rgb(40,187,236)'],
  [0.375, 'rgb(31,233,175)'], [0.5, 'rgb(122,252,82)'], [0.625, 'rgb(217,227,45)'],
  [0.75, 'rgb(252,156,42)'], [0.875, 'rgb(227,68,10)'], [1, 'rgb(122,4,3)'],
];

function detectWebGL(): boolean {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch { return false; }
}
export const HAS_WEBGL = detectWebGL();

function webglNotice(el: HTMLElement, what: string): void {
  el.innerHTML = `<div style="padding:40px;color:#8a8f9c;font-size:13px">
    WebGL is unavailable in this browser, so the ${what} view cannot render.
    Enable hardware acceleration (chrome://settings/system) or use another browser —
    all PSF/incoherence analysis works without it.</div>`;
}

function layout3d(title: string): PlotlyLayout {
  const ax = { ...AXIS, backgroundcolor: '#14161c', showbackground: true };
  return {
    ...DARK, title: { text: title, font: { size: 13 } },
    margin: { l: 0, r: 0, t: 30, b: 0 },
    scene: { xaxis: ax, yaxis: ax, zaxis: ax, aspectmode: 'data' },
    showlegend: false,
  } as PlotlyLayout;
}

/** Sample stride so ilv curves stay drawable. */
function strideFor(nPts: number, target: number): number {
  return Math.max(1, Math.floor(nPts / target));
}

function ilvSamples(t: TrajData, g: number, stride: number): { x: number[]; y: number[]; z: number[] } {
  const irep = Math.floor(g / t.NI), j = g % t.NI;
  const base = irep * t.NI * t.NPTS + j * t.NPTS;
  const x: number[] = [], y: number[] = [], z: number[] = [];
  for (let p = 0; p < t.NPTS; p += stride) {
    x.push(t.kx[base + p]); y.push(t.ky[base + p]); z.push(t.kz[base + p]);
  }
  return { x, y, z };
}

// turbo colormap sample (t in [0,1]) via anchor interpolation
const TURBO_ANCHORS: [number, number, number][] = [
  [48, 18, 59], [70, 107, 227], [40, 187, 236], [31, 233, 175],
  [122, 252, 82], [217, 227, 45], [252, 156, 42], [227, 68, 10], [122, 4, 3],
];
function turbo(t: number): string {
  const x = Math.min(Math.max(t, 0), 1) * (TURBO_ANCHORS.length - 1);
  const i = Math.min(Math.floor(x), TURBO_ANCHORS.length - 2);
  const f = x - i;
  const c = TURBO_ANCHORS[i].map((v, k) => Math.round(v + f * (TURBO_ANCHORS[i + 1][k] - v)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** 3D curves, one line per interleave, turbo by index. Caps at 256 curves. */
export function plotCurves3D(el: HTMLElement, t: TrajData, ilvs: number[]): string | null {
  if (!HAS_WEBGL) { webglNotice(el, '3D curves'); return null; }
  let note: string | null = null;
  let shown = ilvs;
  if (ilvs.length > 256) {
    const stride = Math.ceil(ilvs.length / 256);
    shown = ilvs.filter((_, i) => i % stride === 0);
    note = `showing every ${stride}ᵗʰ of ${ilvs.length} interleaves (curve cap 256)`;
  }
  const ptStride = strideFor(t.NPTS, 160);
  const total = t.NI * t.NREPS;
  const traces = shown.map(g => {
    const s = ilvSamples(t, g, ptStride);
    return {
      type: 'scatter3d', mode: 'lines', ...s,
      line: { color: turbo(g / Math.max(total - 1, 1)), width: 2 },
      name: `ilv ${g}`, hoverinfo: 'name',
    } as PlotlyData;
  });
  Plotly.react(el, traces, layout3d('3D interleave curves (physical k, cycles/m — turbo = global interleave index)'), { responsive: true });
  return note;
}

/** 3D point coverage: one scatter trace, 1-2 px points, turbo by ilv index. */
export function plotPoints3D(el: HTMLElement, t: TrajData, ilvs: number[]): string | null {
  if (!HAS_WEBGL) { webglNotice(el, '3D point coverage'); return null; }
  const budget = 180_000;
  const totalPts = ilvs.length * t.NPTS;
  const stride = Math.max(1, Math.ceil(totalPts / budget));
  const x: number[] = [], y: number[] = [], z: number[] = [], cv: number[] = [];
  for (const g of ilvs) {
    const irep = Math.floor(g / t.NI), j = g % t.NI;
    const base = irep * t.NI * t.NPTS + j * t.NPTS;
    for (let p = 0; p < t.NPTS; p += stride) {
      x.push(t.kx[base + p]); y.push(t.ky[base + p]); z.push(t.kz[base + p]); cv.push(g);
    }
  }
  const trace: PlotlyData = {
    type: 'scatter3d', mode: 'markers', x, y, z,
    marker: {
      size: 1.3, opacity: 0.55, color: cv, colorscale: TURBO_SCALE,
      cmin: 0, cmax: t.NI * t.NREPS - 1,
      colorbar: { title: { text: 'ilv' }, thickness: 12, len: 0.6 },
    },
    hoverinfo: 'skip',
  } as PlotlyData;
  Plotly.react(el, [trace], layout3d('k-space point coverage'), { responsive: true });
  return stride > 1 ? `showing every ${stride}ᵗʰ sample (${x.length.toLocaleString()} of ${totalPts.toLocaleString()} points)` : null;
}

/** 2D plane projections (points), xy / xz / yz side by side. */
export function plotProjections(el: HTMLElement, t: TrajData, ilvs: number[]): string | null {
  const budget = HAS_WEBGL ? 120_000 : 24_000;
  const totalPts = ilvs.length * t.NPTS;
  const stride = Math.max(1, Math.ceil(totalPts / budget));
  const ax: number[] = [], ay: number[] = [], az: number[] = [], cv: number[] = [];
  for (const g of ilvs) {
    const irep = Math.floor(g / t.NI), j = g % t.NI;
    const base = irep * t.NI * t.NPTS + j * t.NPTS;
    for (let p = 0; p < t.NPTS; p += stride) {
      ax.push(t.kx[base + p]); ay.push(t.ky[base + p]); az.push(t.kz[base + p]); cv.push(g);
    }
  }
  const planes: [string, number[], number[]][] = [['xy', ax, ay], ['xz', ax, az], ['yz', ay, az]];
  const traces: PlotlyData[] = planes.map(([name, px, py], i) => ({
    type: HAS_WEBGL ? 'scattergl' : 'scatter', mode: 'markers', x: px, y: py,
    xaxis: `x${i + 1}`, yaxis: `y${i + 1}`,
    marker: { size: 1.5, opacity: 0.4, color: cv, colorscale: TURBO_SCALE, cmin: 0, cmax: t.NI * t.NREPS - 1 },
    name, hoverinfo: 'skip', showlegend: false,
  } as PlotlyData));
  const axc = { ...AXIS, scaleanchor: undefined };
  Plotly.react(el, traces, {
    ...DARK, grid: { rows: 1, columns: 3, pattern: 'independent' },
    margin: { l: 45, r: 10, t: 40, b: 40 },
    title: { text: 'plane projections (sample points)', font: { size: 13 } },
    xaxis: { ...axc, title: { text: 'kx' } }, yaxis: { ...axc, title: { text: 'ky' }, scaleanchor: 'x' },
    xaxis2: { ...axc, title: { text: 'kx' } }, yaxis2: { ...axc, title: { text: 'kz' }, scaleanchor: 'x2' },
    xaxis3: { ...axc, title: { text: 'ky' } }, yaxis3: { ...axc, title: { text: 'kz' }, scaleanchor: 'x3' },
  } as PlotlyLayout, { responsive: true });
  return stride > 1 ? `showing every ${stride}ᵗʰ sample` : null;
}

function centralSlice(mag: Float32Array, n: number): number[][] {
  // slice at x = c (matches reference aal[c]): rows y, cols z
  const c = n >> 1;
  const rows: number[][] = [];
  for (let y = 0; y < n; y++) {
    const row: number[] = [];
    for (let z = 0; z < n; z++) row.push(Math.log10(mag[(c * n + y) * n + z] + 1e-5));
    rows.push(row);
  }
  return rows;
}

/** |PSF| and |alias| central slices, log10, magma. */
export function plotPsfSlices(el: HTMLElement, r: PsfResult): void {
  const magP = absVol(r.psfRe, r.psfIm);
  const magA = absVol(r.aliasRe, r.aliasIm);
  const traces: PlotlyData[] = [
    { type: 'heatmap', z: centralSlice(magP, r.n), colorscale: MAGMA, zmin: -4, zmax: 0,
      colorbar: { title: { text: 'log₁₀|PSF|' }, thickness: 12, len: 0.8 }, xaxis: 'x', yaxis: 'y' } as PlotlyData,
    { type: 'heatmap', z: centralSlice(magA, r.n), colorscale: MAGMA, zmin: -4, zmax: -0.5,
      colorbar: { title: { text: 'log₁₀|alias|' }, thickness: 12, len: 0.8, x: 1.0 }, xaxis: 'x2', yaxis: 'y2' } as PlotlyData,
  ];
  Plotly.react(el, traces, {
    ...DARK, grid: { rows: 1, columns: 2, pattern: 'independent' },
    margin: { l: 40, r: 10, t: 40, b: 35 },
    title: { text: `central slice — |PSF| (left) and |alias = PSF − PSF_full| (right), ${r.label}`, font: { size: 13 } },
    xaxis: { ...AXIS, constrain: 'domain' }, yaxis: { ...AXIS, scaleanchor: 'x' },
    xaxis2: { ...AXIS, constrain: 'domain' }, yaxis2: { ...AXIS, scaleanchor: 'x2' },
  } as PlotlyLayout, { responsive: true });
}

/** Radial shell mean/max of |PSF| vs full reference (reference col 1). */
export function plotShells(el: HTMLElement, r: PsfResult): void {
  const mk = (xs: Float64Array, ys: Float64Array, name: string, color: string, dash?: 'dash' | 'dot'): PlotlyData => ({
    type: 'scatter', mode: 'lines', x: [...xs], y: [...ys],
    name, line: { color, width: 1.6, dash },
  } as PlotlyData);
  const traces = [
    mk(r.shell.r, r.shell.max, 'ensemble shell max', '#e0526b'),
    mk(r.shell.r, r.shell.mean, 'ensemble shell mean', '#5b8fd9'),
    mk(r.shellFull.r, r.shellFull.max, 'full-set shell max', '#e8e9ee'),
    mk(r.shellFull.r, r.shellFull.mean, 'full-set shell mean', '#8a8f9c', 'dash'),
  ];
  Plotly.react(el, traces, {
    ...DARK, margin: { l: 55, r: 10, t: 40, b: 40 },
    title: { text: `|PSF| radial shells — FWHM ${r.fwhm.toFixed(2)} vox (full ${r.fwhmFull.toFixed(2)})`, font: { size: 13 } },
    xaxis: { ...AXIS, title: { text: 'radius (vox)' } },
    yaxis: { ...AXIS, type: 'log', title: { text: '|PSF|' }, exponentformat: 'power' },
    legend: { x: 0.62, y: 0.98, bgcolor: 'rgba(20,22,28,0.7)' },
  } as PlotlyLayout, { responsive: true });
}

/** Alias field radial curves + noise-scale lines (reference col 2). */
export function plotAliasRadial(el: HTMLElement, r: PsfResult): void {
  const s = r.aliasStats;
  const mk = (xs: number[], ys: number[], name: string, color: string, dash?: 'dash' | 'dot', mode = 'lines'): PlotlyData => ({
    type: 'scatter', mode, x: xs, y: ys, name, line: { color, width: 1.6, dash },
  } as PlotlyData);
  const xr = [...r.aliasShell.r];
  const traces = [
    mk(xr, [...r.aliasShell.max], 'alias shell max', '#e0526b'),
    mk(xr, [...r.aliasShell.mean], 'alias shell mean', '#5b8fd9'),
    mk([xr[0], xr[xr.length - 1]], [s.sigma, s.sigma], `σ = ${s.sigma.toFixed(4)}`, '#8a8f9c', 'dash'),
    mk([xr[0], xr[xr.length - 1]], [s.expMax, s.expMax], `noise-like max = ${s.expMax.toFixed(4)}`, '#4fae62', 'dot'),
    { type: 'scatter', mode: 'markers', x: [s.peakR], y: [s.max], name: `alias max ${s.max.toFixed(4)}`,
      marker: { color: '#e0526b', size: 9, symbol: 'circle-open' } } as PlotlyData,
  ];
  Plotly.react(el, traces, {
    ...DARK, margin: { l: 55, r: 10, t: 40, b: 40 },
    title: { text: `alias field — max/noise-like = ${s.coh.toFixed(2)}, kurtosis ${s.kurt.toFixed(2)}`, font: { size: 13 } },
    xaxis: { ...AXIS, title: { text: 'radius (vox)' } },
    yaxis: { ...AXIS, type: 'log', title: { text: '|alias|' }, exponentformat: 'power' },
    legend: { x: 0.58, y: 0.02, bgcolor: 'rgba(20,22,28,0.7)' },
  } as PlotlyLayout, { responsive: true });
}

/** Fan views: rows full / ensemble / alias × cols xy / xz / yz.
 *  Fixed log scale vmin -3.5 vmax -0.5 (F11). */
export function plotFans(el: HTMLElement, r: PsfResult): void {
  const fields: [string, Float32Array][] = [
    ['full set', absVol(r.fullRe, r.fullIm)],
    [r.label, absVol(r.psfRe, r.psfIm)],
    ['alias', absVol(r.aliasRe, r.aliasIm)],
  ];
  const planes: ('xy' | 'xz' | 'yz')[] = ['xy', 'xz', 'yz'];
  const traces: PlotlyData[] = [];
  let idx = 1;
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const { rs, fan, nTheta } = fanView(fields[row][1], r.n, planes[col]);
      const nr = rs.length;
      const z: number[][] = [];
      for (let t = 0; t < nTheta; t++) {
        const rowv: number[] = [];
        for (let i = 0; i < nr; i++) rowv.push(Math.log10(fan[t * nr + i] + 1e-5));
        z.push(rowv);
      }
      traces.push({
        type: 'heatmap', z, x: [...rs], y: Array.from({ length: nTheta }, (_, t) => t),
        colorscale: MAGMA, zmin: -3.5, zmax: -0.5, showscale: idx === 9,
        colorbar: { title: { text: 'log₁₀|PSF|' }, thickness: 12, len: 0.9 },
        xaxis: `x${idx}`, yaxis: `y${idx}`,
      } as PlotlyData);
      idx++;
    }
  }
  const lay: Record<string, unknown> = {
    ...DARK, grid: { rows: 3, columns: 3, pattern: 'independent' },
    margin: { l: 55, r: 10, t: 60, b: 40 },
    title: { text: 'fan views |PSF|(r, θ) — rings → vertical stripes, directional alias → horizontal streaks<br><sub>rows: full set / ensemble / alias · cols: xy, xz, yz central planes · fixed log scale [−3.5, −0.5]</sub>', font: { size: 13 } },
  };
  const rowNames = ['full', 'ens', 'alias'];
  for (let i = 1; i <= 9; i++) {
    const row = Math.floor((i - 1) / 3), col = (i - 1) % 3;
    lay[`xaxis${i === 1 ? '' : i}`] = { ...AXIS, title: row === 2 ? { text: 'radius (vox)' } : undefined };
    lay[`yaxis${i === 1 ? '' : i}`] = { ...AXIS, title: col === 0 ? { text: `${rowNames[row]} · θ (deg)` } : undefined };
  }
  Plotly.react(el, traces, lay as PlotlyLayout, { responsive: true });
}

export function purge(el: HTMLElement): void {
  Plotly.purge(el);
}
