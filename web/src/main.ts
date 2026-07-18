// Main thread: UI wiring, worker protocol, plot orchestration.

import type { GenParams, TrajData, PsfResult, WorkerResponse, GoldenReport } from './types';
import { V3_DEFAULTS } from './types';
import { parseEnsemble, PRESETS } from './ensemble';
import * as plots from './plots';
import batteryPng from './assets/alias_psf_battery.png';
import fanPng from './assets/psf_fan_view.png';

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

let nextId = 1;
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();

function request<T>(op: 'generate' | 'psf' | 'golden', payload: object = {}): Promise<T> {
  const id = nextId++;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    worker.postMessage({ id, op, ...payload });
  });
}

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const statusEl = $('status');

function setStatus(msg: string, cls: '' | 'busy' | 'error' = '') {
  statusEl.textContent = msg;
  statusEl.className = cls;
}

worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
  const m = ev.data;
  if (m.op === 'progress') { setStatus(m.msg, 'busy'); return; }
  const p = pending.get(m.id);
  if (!p) return;
  pending.delete(m.id);
  if (m.op === 'error') { setStatus(m.msg, 'error'); p.reject(new Error(m.msg)); }
  else if (m.op === 'generate') p.resolve(m.traj);
  else if (m.op === 'psf') p.resolve(m.result);
  else if (m.op === 'golden') p.resolve(m.report);
};

// ---------------------------------------------------------------- state

let traj: TrajData | null = null;
let lastParams: GenParams = V3_DEFAULTS;
let lastPsf: PsfResult | null = null;
const compare: PsfResult[] = [];

// ---------------------------------------------------------------- params

function readParams(): GenParams {
  const num = (id: string) => parseFloat(($(id) as HTMLInputElement).value);
  const axis = ($('p-axis') as HTMLInputElement).value.split(',').map(s => parseFloat(s.trim()));
  return {
    NI: num('p-NI') | 0, NPTS: num('p-NPTS') | 0, NREPS: num('p-NREPS') | 0,
    n: num('p-n'), at: num('p-at') * 1e-3, fov: num('p-fov'), ms: num('p-ms'),
    optimize: ($('p-opt') as HTMLInputElement).checked,
    rotAngleDeg: num('p-rot'),
    axisMode: ($('p-axismode') as HTMLSelectElement).value as 'hedgehog' | 'fixed',
    axis: [axis[0] || 0, axis[1] || 0, axis[2] || 1],
    t0: num('p-t0') * 1e-6, dt: num('p-dt') * 1e-6,
    maxs: num('p-maxs'), maxg: num('p-maxg'),
  };
}

$('p-rot').addEventListener('input', () => {
  $('rotAngleVal').textContent = `${($('p-rot') as HTMLInputElement).value}°`;
});
$('p-axismode').addEventListener('change', () => {
  $('fixedaxis-row').style.display =
    ($('p-axismode') as HTMLSelectElement).value === 'fixed' ? 'flex' : 'none';
});

// ---------------------------------------------------------------- ensemble

const presetSel = $('ens-preset') as HTMLSelectElement;
presetSel.innerHTML = '<option value="">presets…</option>' +
  PRESETS.map((p, i) => `<option value="${i}">${p.label}</option>`).join('');
presetSel.addEventListener('change', () => {
  if (presetSel.value === '') return;
  const p = PRESETS[parseInt(presetSel.value, 10)];
  const NI = traj?.NI ?? V3_DEFAULTS.NI, NREPS = traj?.NREPS ?? V3_DEFAULTS.NREPS;
  ($('ens-expr') as HTMLInputElement).value = p.expr(NI, NREPS);
  presetSel.value = '';
  updateViews();
});

function colorMode(): plots.ColorMode {
  return ($('color-mode') as HTMLSelectElement).value as plots.ColorMode;
}

function currentEnsemble(): number[] {
  if (!traj) throw new Error('generate first');
  return parseEnsemble(($('ens-expr') as HTMLInputElement).value, traj.NI, traj.NREPS);
}

// ---------------------------------------------------------------- views

const VIEWS = ['curves', 'points', 'proj', 'psf', 'fan', 'fountain', 'poly', 'compare', 'about'];
let activeView = 'curves';
for (const btn of $('tabs').querySelectorAll('button')) {
  btn.addEventListener('click', () => {
    activeView = btn.dataset.view!;
    for (const b of $('tabs').querySelectorAll('button')) b.classList.toggle('active', b === btn);
    for (const v of VIEWS) $(`view-${v}`).classList.toggle('active', v === activeView);
    renderActive();
  });
}

const rendered = new Set<string>();

function renderActive() {
  if (!traj) return;
  try {
    const ilvs = currentEnsemble();
    if (activeView === 'curves' && !rendered.has('curves')) {
      const note = plots.plotCurves3D($('plot-curves'), traj, ilvs, colorMode());
      $('note-curves').textContent = note ?? '';
      rendered.add('curves');
    } else if (activeView === 'points' && !rendered.has('points')) {
      const note = plots.plotPoints3D($('plot-points'), traj, ilvs, colorMode());
      $('note-points').textContent = note ?? '';
      rendered.add('points');
    } else if (activeView === 'proj' && !rendered.has('proj')) {
      const note = plots.plotProjections($('plot-proj'), traj, ilvs, colorMode());
      $('note-proj').textContent = note ?? '';
      rendered.add('proj');
    } else if (activeView === 'psf' && lastPsf && !rendered.has('psf')) {
      plots.plotPsfSlices($('plot-slices'), lastPsf);
      plots.plotShells($('plot-shells'), lastPsf);
      plots.plotAliasRadial($('plot-alias'), lastPsf);
      $('metrics-box').innerHTML = metricsTable([lastPsf]);
      rendered.add('psf');
    } else if (activeView === 'fan' && lastPsf && !rendered.has('fan')) {
      plots.plotFans($('plot-fan'), lastPsf);
      rendered.add('fan');
    } else if (activeView === 'poly' && !rendered.has('poly')) {
      renderPolyhedron();
      rendered.add('poly');
    } else if (activeView === 'fountain' && lastPsf && !rendered.has('fountain')) {
      plots.plotFountain($('plot-fountain'), lastPsf,
        ($('fountain-plane') as HTMLSelectElement).value as 'xy' | 'xz' | 'yz',
        ($('fountain-field') as HTMLSelectElement).value as 'ens' | 'full' | 'alias');
      rendered.add('fountain');
    }
  } catch (e) {
    setStatus(e instanceof Error ? e.message : String(e), 'error');
  }
}

function invalidateViews(alsoP = false) {
  rendered.delete('curves'); rendered.delete('points'); rendered.delete('proj');
  rendered.delete('poly');
  if (alsoP) { rendered.delete('psf'); rendered.delete('fan'); }
}

function updateViews() {
  invalidateViews();
  renderActive();
}

$('btn-view').addEventListener('click', updateViews);
$('color-mode').addEventListener('change', updateViews);
$('poly-which').addEventListener('change', () => { rendered.delete('poly'); renderActive(); });
for (const id of ['fountain-field', 'fountain-plane'])
  $(id).addEventListener('change', () => { rendered.delete('fountain'); renderActive(); });
$('ens-expr').addEventListener('keydown', (e) => { if ((e as KeyboardEvent).key === 'Enter') updateViews(); });

// ---------------------------------------------------------------- polyhedron

// Known Thomson-problem optimal configurations (global minima) by point count.
const THOMSON_NAMES: Record<number, string> = {
  2: 'antipodal pair',
  3: 'equilateral triangle (equatorial)',
  4: 'regular tetrahedron',
  5: 'triangular bipyramid',
  6: 'regular octahedron',
  7: 'pentagonal bipyramid',
  8: 'square antiprism (NOT a cube!)',
  9: 'triaugmented triangular prism',
  10: 'gyroelongated square bipyramid',
  11: 'edge-contracted icosahedron',
  12: 'regular icosahedron',
  24: 'snub cube',
  32: 'pentakis dodecahedron (icosahedron + dodecahedron vertices)',
};

function minPairAngleDeg(pts: Float32Array, count: number): number {
  let maxDot = -2;
  for (let i = 0; i < count; i++)
    for (let j = i + 1; j < count; j++) {
      const d = pts[3*i]*pts[3*j] + pts[3*i+1]*pts[3*j+1] + pts[3*i+2]*pts[3*j+2];
      if (d > maxDot) maxDot = d;
    }
  return (Math.acos(Math.min(1, Math.max(-1, maxDot))) * 180) / Math.PI;
}

function renderPolyhedron() {
  if (!traj) return;
  const which = ($('poly-which') as HTMLSelectElement).value;
  const pts = which === 'basis' ? traj.basis : traj.reprot;
  const count = which === 'basis' ? traj.NI : traj.NREPS;
  const what = which === 'basis' ? 'charge basis' : 'rotation-axis hedgehog';
  const named = THOMSON_NAMES[count];
  const nameTxt = lastParams.optimize
    ? (named ? `N = ${count}: <b style="color:#e8e9ee">${named}</b> (Thomson optimum)`
             : `N = ${count}: generic Thomson configuration — no named polyhedron`)
    : `N = ${count}: Fibonacci spiral (Thomson optimization OFF — names apply only to optimized configs)`;
  const ang = count > 1 ? ` · min pairwise angle ${minPairAngleDeg(pts, count).toFixed(1)}°` : '';
  $('poly-name').innerHTML = nameTxt + ang;
  plots.plotPolyhedron($('plot-poly'), pts, count,
    `convex hull of the ${what} (${count} unit vectors)`);
}

// ---------------------------------------------------------------- metrics table

function fmt(v: number, d = 3) { return Number.isFinite(v) ? v.toFixed(d) : '—'; }

function metricsTable(rs: PsfResult[]): string {
  const rows: [string, (r: PsfResult) => string, boolean?][] = [
    ['ensemble', r => `${r.label} (${r.nIlv} ilv, ${r.nSamples.toLocaleString()} samples)`],
    ['image size N', r => `${r.n}`],
    ['FWHM (vox)', r => fmt(r.fwhm, 2)],
    ['FWHM full set (vox)', r => fmt(r.fwhmFull, 2)],
    ['alias max', r => fmt(r.aliasStats.max, 4)],
    ['alias σ (Rayleigh)', r => fmt(r.aliasStats.sigma, 5)],
    ['noise-like max σ√(2 ln n)', r => fmt(r.aliasStats.expMax, 4)],
    ['max / noise-like max', r => fmt(r.aliasStats.coh, 2), true],
    ['excess kurtosis (re/im)', r => fmt(r.aliasStats.kurt, 2)],
    ['alias peak r (vox)', r => fmt(r.aliasStats.peakR, 1)],
    ['alias peak direction', r => r.aliasStats.peakDir.map(v => fmt(v, 2)).join(', ')],
    ['dipole |⟨k̂⟩| (|k|>0.2 kmax)', r => fmt(r.dipole, 4)],
  ];
  let h = '<table class="metrics"><tr><th>metric</th>' + rs.map((_, i) => `<th>#${i + 1}</th>`).join('') + '</tr>';
  for (const [name, get, headline] of rows)
    h += `<tr><td class="name">${name}</td>` +
         rs.map(r => `<td class="${headline ? 'headline' : ''}">${get(r)}</td>`).join('') + '</tr>';
  h += '</table><div class="note">max/noise-like = 1.0 → perfectly incoherent (noise-like) aliasing; ' +
       'v3 aligned windows measure ≈ 1.5. σ is Rayleigh-correct: √(E|x|²/2) over r &gt; 2·FWHM.</div>';
  return h;
}

// ---------------------------------------------------------------- actions

$('btn-generate').addEventListener('click', async () => {
  const params = readParams();
  setStatus('generating trajectory…', 'busy');
  ($('btn-generate') as HTMLButtonElement).disabled = true;
  try {
    traj = await request<TrajData>('generate', { params });
    lastParams = params;
    lastPsf = null;
    rendered.clear();
    const totalIlv = traj.NI * traj.NREPS;
    $('gen-info').innerHTML =
      `${totalIlv} interleaves (${traj.NI} × ${traj.NREPS}), ${traj.total.toLocaleString()} samples, kmax ${traj.kmax.toFixed(1)} c/m` +
      (traj.nanCount ? `<br><span style="color:#e0526b">⚠ ${traj.nanCount.toLocaleString()} samples violate slew/gradient limits ` +
        `(NaN — dropped from PSF). Lower n toward 2, lengthen readout, or raise limits.</span>` : '');
    setStatus(`trajectory ready — ${totalIlv} interleaves`);
    renderActive();
  } catch { /* status already set */ }
  ($('btn-generate') as HTMLButtonElement).disabled = false;
});

$('btn-psf').addEventListener('click', async () => {
  if (!traj) { setStatus('generate a trajectory first', 'error'); return; }
  let ilvs: number[];
  try { ilvs = currentEnsemble(); } catch (e) { setStatus(String(e), 'error'); return; }
  if (!ilvs.length) { setStatus('empty ensemble', 'error'); return; }
  const n = parseInt(($('p-psfn') as HTMLSelectElement).value, 10);
  const label = ($('ens-expr') as HTMLInputElement).value.trim();
  ($('btn-psf') as HTMLButtonElement).disabled = true;
  try {
    lastPsf = await request<PsfResult>('psf', { ilvs, n, label });
    rendered.delete('psf'); rendered.delete('fan'); rendered.delete('fountain');
    ($('btn-compare') as HTMLButtonElement).disabled = false;
    setStatus(`PSF done — max/noise-like = ${lastPsf.aliasStats.coh.toFixed(2)}`);
    if (activeView !== 'psf' && activeView !== 'fan') {
      ($('tabs').querySelector('[data-view=psf]') as HTMLButtonElement).click();
    } else renderActive();
  } catch { /* status set */ }
  ($('btn-psf') as HTMLButtonElement).disabled = false;
});

$('btn-compare').addEventListener('click', () => {
  if (!lastPsf) return;
  if (compare.length >= 4) compare.shift();
  compare.push(lastPsf);
  renderCompare();
  ($('tabs').querySelector('[data-view=compare]') as HTMLButtonElement).click();
});

function renderCompare() {
  const box = $('compare-box');
  if (!compare.length) return;
  box.innerHTML = metricsTable(compare) +
    '<button id="btn-clear-compare" class="secondary">clear</button>';
  $('btn-clear-compare').addEventListener('click', () => {
    compare.length = 0;
    box.innerHTML = '<p class="note">cleared.</p>';
  });
}

// ---------------------------------------------------------------- about

function aboutHtml(): string {
  return `
  <h2>What this is</h2>
  <p>Interactive explorer for ASAP ("bent spiral") k-space trajectories and their incoherence
  properties. The trajectory generator is the original <code>kasap.c</code> compiled to WebAssembly;
  the PSF machinery is a gridding NUFFT (Kaiser–Bessel, 2× oversampling, Beatty β) validated
  against FINUFFT to 4·10⁻⁴ max relative error on the golden reference PSFs.</p>

  <h2>Conventions</h2>
  <ul>
    <li>Adjoint: f(x) = Σ cⱼ e⁻ⁱωx; forward: cⱼ = Σ f(x) eⁱωx. DCF: w = 1/clip(|A Aᴴ 1|, max·10⁻⁴), mean-normalized.</li>
    <li>PSF = Aᴴ·DCF·A·δ, complex-normalized at center. Alias = PSF<sub>ensemble</sub> − PSF<sub>full</sub> (aperture/Gibbs ring cancels).</li>
    <li>σ = √(E|x|²/2) over r &gt; 2·FWHM (Rayleigh-correct); headline metric = max/(σ√(2 ln n)); 1.0 = perfectly incoherent.</li>
    <li>Generated trajectories: ω = 2π·k·FOV/ms (band edge π at design kmax). The golden dump uses ω = (g−120)·2π/240.</li>
  </ul>

  <h2>Golden validation</h2>
  <p>The golden reference is the <b>real v3 832×510 measured trajectory</b> (26 Thomson charges ×
  32 rotations). Note it is not bit-reproducible by the generator defaults — it is the measured
  scanner trajectory (different radial shape, per-interleave distortions) — so validation runs the
  app's NUFFT/PSF chain on the golden trajectory itself and compares against the reference numbers
  from the FINUFFT implementation.</p>
  <p><button id="btn-golden">Run golden validation (in-browser, ~1 min)</button></p>
  <div id="golden-result"></div>
  <h2>Reference figures (ground truth, XeCS session 2026-07-17)</h2>
  <p><img src="${batteryPng}" alt="alias PSF battery reference"></p>
  <p><img src="${fanPng}" alt="PSF fan view reference"></p>
  <p class="note">Reading the fan views: rings appear as vertical stripes, directional aliasing as
  horizontal streaks. Fixed log₁₀ color scale [−3.5, −0.5] for comparability.</p>`;
}

const EXPECT = {
  fullFwhm: 2.188, fullSideMax: 0.0302,
  b0Fwhm: 2.460, b0AliasMax: 0.1164, b0Sigma: 0.0160, b0Coh: 1.458,
};

function goldenTable(r: GoldenReport): string {
  const row = (name: string, got: number, want: number, d = 3) => {
    const ok = Math.abs(got - want) / want < 0.10;
    return `<tr><td class="name">${name}</td><td>${got.toFixed(d)}</td><td>${want.toFixed(d)}</td>
      <td style="color:${ok ? '#4fae62' : '#e0526b'}">${ok ? 'PASS' : 'FAIL'} (${(100 * (got - want) / want).toFixed(1)}%)</td></tr>`;
  };
  return `<table class="metrics"><tr><th>metric</th><th>app</th><th>reference</th><th>Δ (tol ±10%)</th></tr>` +
    row('full-832 FWHM (vox)', r.fullFwhm, EXPECT.fullFwhm) +
    row('full-832 aperture-ring peak', r.fullSideMax, EXPECT.fullSideMax, 4) +
    row('block-0 FWHM (vox)', r.b0Fwhm, EXPECT.b0Fwhm) +
    row('block-0 alias max', r.b0AliasMax, EXPECT.b0AliasMax, 4) +
    row('block-0 alias σ', r.b0Sigma, EXPECT.b0Sigma, 4) +
    row('block-0 max/noise-like', r.b0Coh, EXPECT.b0Coh) +
    '</table>';
}

$('about').innerHTML = aboutHtml();
$('btn-golden').addEventListener('click', async () => {
  ($('btn-golden') as HTMLButtonElement).disabled = true;
  try {
    const r = await request<GoldenReport>('golden');
    $('golden-result').innerHTML = goldenTable(r);
    setStatus('golden validation done');
  } catch { /* status set */ }
  ($('btn-golden') as HTMLButtonElement).disabled = false;
});

// ---------------------------------------------------------------- boot

setStatus('generating default v3 trajectory…', 'busy');
(async () => {
  try {
    traj = await request<TrajData>('generate', { params: V3_DEFAULTS });
    $('gen-info').textContent =
      `${traj.NI * traj.NREPS} interleaves (${traj.NI} × ${traj.NREPS}), ${traj.total.toLocaleString()} samples, kmax ${traj.kmax.toFixed(1)} c/m`;
    setStatus('ready — v3 defaults loaded');
    renderActive();
  } catch { /* status set */ }
})();
