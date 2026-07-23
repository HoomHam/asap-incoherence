// Main thread: UI wiring, worker protocol, plot orchestration.

import type { GenParams, TrajData, PsfResult, WorkerResponse, GoldenReport } from './types';
import { V3_DEFAULTS } from './types';
import { parseEnsemble, PRESETS } from './ensemble';
import { shellDensity } from './metrics';
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

const VIEWS = ['curves', 'proj', 'psf', 'fan', 'fountain', 'poly', 'compare', 'about'];
let activeView = 'poly';
for (const btn of $('tabs').querySelectorAll('button')) {
  btn.addEventListener('click', () => {
    if (activeView === 'poly' && btn.dataset.view !== 'poly') stopPolyAnim();
    if (activeView === 'curves' && btn.dataset.view !== 'curves' && curveAnim.running()) { curveAnim.stop(); rendered.delete('curves'); }
    if (activeView === 'proj' && btn.dataset.view !== 'proj' && projAnim.running()) { projAnim.stop(); rendered.delete('proj'); }
    activeView = btn.dataset.view!;
    for (const b of $('tabs').querySelectorAll('button')) b.classList.toggle('active', b === btn);
    for (const v of VIEWS) $(`view-${v}`).classList.toggle('active', v === activeView);
    renderActive();
  });
}

const rendered = new Set<string>();

const allIlvs = () => Array.from({ length: traj ? traj.NI * traj.NREPS : 0 }, (_, i) => i);

function renderActive() {
  if (!traj) return;
  try {
    const ilvs = currentEnsemble();
    if (activeView === 'curves' && !rendered.has('curves')) {
      const note = plots.plotCurves3D($('plot-curves'), traj, ilvs, colorMode());
      $('note-curves').textContent = note ?? '';
      rendered.add('curves');
    } else if (activeView === 'proj' && !rendered.has('proj')) {
      const ensD = shellDensity(traj, ilvs);
      const fullD = shellDensity(traj, allIlvs());
      const note = plots.plotProjections($('plot-proj'), traj, ilvs, colorMode(),
        undefined, { rEns: ensD.rN, rFull: fullD.rN });
      $('note-proj').textContent =
        `${note ? note + ' · ' : ''}r_N ${ensD.rN.toFixed(0)} (full ${fullD.rN.toFixed(0)}) of kmax ${ensD.kmaxGrid.toFixed(0)} grid units`;
      plots.plotShellDensity($('plot-density'), ensD, fullD,
        ($('ens-expr') as HTMLInputElement).value.trim() || 'all');
      rendered.add('proj');
    } else if (activeView === 'psf' && lastPsf && !rendered.has('psf')) {
      renderSlices();
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
  rendered.delete('curves'); rendered.delete('proj');
  rendered.delete('poly');
  if (alsoP) { rendered.delete('psf'); rendered.delete('fan'); }
}

function updateViews() {
  curveAnim.stop();
  projAnim.stop();
  invalidateViews();
  renderActive();
}

$('btn-view').addEventListener('click', updateViews);
$('color-mode').addEventListener('change', updateViews);
$('poly-which').addEventListener('change', () => {
  ($('poly-animate') as HTMLButtonElement).disabled = ($('poly-which') as HTMLSelectElement).value !== 'basis';
  rendered.delete('poly'); renderActive();
});
$('poly-animate').addEventListener('click', () => {
  if (polyAnimId !== null) { stopPolyAnim(); rendered.delete('poly'); renderActive(); }
  else startPolyAnim();
});
$('curves-animate').addEventListener('click', () => {
  if (curveAnim.running()) { curveAnim.stop(); rendered.delete('curves'); renderActive(); }
  else curveAnim.start();
});
$('proj-animate').addEventListener('click', () => {
  if (projAnim.running()) { projAnim.stop(); rendered.delete('proj'); renderActive(); }
  else projAnim.start();
});
for (const id of ['fountain-field', 'fountain-plane'])
  $(id).addEventListener('change', () => { rendered.delete('fountain'); renderActive(); });

// ---------------------------------------------------------------- PSF slice browser
function positionCenterTick() {
  // mark the central-slice position; thumb travel is (width − thumbW), thumb ≈ 16px
  const slider = $('slice-idx') as HTMLInputElement;
  const max = parseInt(slider.max, 10) || 1;
  const frac = ((max + 1) >> 1) / max;
  const w = slider.offsetWidth;
  if (w > 0) $('slice-center').style.left = `${8 + frac * (w - 16) - 1}px`;
}
window.addEventListener('resize', positionCenterTick);

function renderSlices() {
  positionCenterTick();
  if (!lastPsf) return;
  const plane = ($('slice-plane') as HTMLSelectElement).value as plots.SlicePlane;
  const slider = $('slice-idx') as HTMLInputElement;
  const idx = Math.min(lastPsf.n - 1, parseInt(slider.value, 10) || 0);
  const perp = plane === 'yz' ? 'x' : plane === 'xz' ? 'y' : 'z';
  $('slice-val').textContent = `${perp} = ${idx} / ${lastPsf.n - 1}`;
  plots.plotPsfSlices($('plot-slices'), lastPsf, plane, idx);
}
$('slice-plane').addEventListener('change', renderSlices);
$('slice-idx').addEventListener('input', renderSlices);
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

// --- readout animations (3D curves + projections): draw every selected
// interleave sample by sample, k-space center -> kmax, over one (scaled)
// readout duration.
function makeReadoutAnim(btnId: string, frameId: string, noteId: string, viewKey: string,
                         needsWebgl: boolean,
                         render: (upTo: number) => string | null,
                         seqCount?: () => number) {
  let id: number | null = null;
  const stop = () => {
    if (id !== null) { cancelAnimationFrame(id); id = null; }
    $(btnId).textContent = '▶ draw readout';
    $(frameId).textContent = '';
  };
  const start = () => {
    if (!traj) return;
    if (needsWebgl && !plots.HAS_WEBGL) { setStatus('animation needs WebGL (hardware acceleration)', 'error'); return; }
    const t = traj;
    // one readout, same at-proportional time base as the sequence animation
    const DUR_MS = Math.min(20000, Math.max(1000, 6000 * lastParams.at / V3_DEFAULTS.at));
    // sequential mode: interleaves fire one by one — per-ilv time shrinks as
    // 1/sqrt(n) so a full block stays watchable; whole run capped at 30 s
    const nSeq = seqCount ? seqCount() : 1;
    const totalMs = seqCount ? Math.min(30000, DUR_MS * Math.sqrt(nSeq)) : DUR_MS;
    const totalSamples = nSeq * t.NPTS;
    const t0 = performance.now();
    const step = () => {
      const ph = Math.min(1, (performance.now() - t0) / totalMs);
      const upTo = Math.max(2, Math.ceil(ph * totalSamples));
      const note = render(upTo);
      const cur = Math.min(nSeq - 1, Math.floor((upTo - 1) / t.NPTS));
      $(frameId).textContent = seqCount
        ? `ilv ${cur + 1}/${nSeq} · sample ${upTo - cur * t.NPTS}/${t.NPTS}`
        : `t ${(ph * lastParams.at * 1e3).toFixed(2)} ms · sample ${upTo}/${t.NPTS}`;
      if (ph >= 1) {
        stop();
        $(frameId).textContent = seqCount
          ? `done — ${nSeq} interleaves fired, ${t.NPTS} samples each`
          : `done — full readout, ${t.NPTS} samples`;
        $(noteId).textContent = note ?? '';
        rendered.add(viewKey);
        return;
      }
      id = requestAnimationFrame(step);
    };
    $(btnId).textContent = '⏸ stop';
    id = requestAnimationFrame(step);
  };
  return { stop, start, running: () => id !== null };
}

// curves animation caps at 256 drawn curves (stride) — sequence over what's shown
const shownCurveCount = () => {
  const n = currentEnsemble().length;
  return n > 256 ? Math.ceil(n / Math.ceil(n / 256)) : n;
};
const curveAnim = makeReadoutAnim('curves-animate', 'curves-frame', 'note-curves', 'curves', true,
  (upTo) => plots.plotCurves3D($('plot-curves'), traj!, currentEnsemble(), colorMode(), upTo, true),
  shownCurveCount);
const projAnim = makeReadoutAnim('proj-animate', 'proj-frame', 'note-proj', 'proj', false,
  (upTo) => plots.plotProjections($('plot-proj'), traj!, currentEnsemble(), colorMode(), upTo));

// --- polyhedron sequence animation: replay the actual k(t) samples rep by
// rep — orientation snaps between reps (cumulative, as kasap applies it)
// and the path bends like the real spiral.
let polyAnimId: number | null = null;

const polySpeed = () => Math.pow(2, parseFloat(($('poly-speed') as HTMLInputElement).value));
$('poly-speed').addEventListener('input', () => {
  const s = polySpeed();
  $('poly-speed-val').textContent = `${parseFloat(s.toFixed(2))}×`;
});
const polyTrail = () => parseInt(($('poly-trail') as HTMLInputElement).value, 10) || 18;
$('poly-trail').addEventListener('input', () => {
  $('poly-trail-val').textContent = `${polyTrail()}`;
});
const polyZoom = () => Math.pow(2, parseFloat(($('poly-zoom') as HTMLInputElement).value));
$('poly-zoom').addEventListener('input', () => {
  $('poly-zoom-val').textContent = `${parseFloat(polyZoom().toFixed(2))}×`;
});

function stopPolyAnim() {
  if (polyAnimId !== null) { cancelAnimationFrame(polyAnimId); polyAnimId = null; }
  $('poly-animate').textContent = '▶ animate sequence';
  $('poly-frame').textContent = '';
}

function startPolyAnim() {
  if (!traj) return;
  if (!plots.HAS_WEBGL) { setStatus('animation needs WebGL (hardware acceleration)', 'error'); return; }
  const t = traj;
  // one readout only (rep 0, no rotations): 6 s at the v3 default readout
  // (5.12 ms); other readouts scale proportionally. Clamped to stay watchable.
  const REP_MS = Math.min(20000, Math.max(1000, 6000 * lastParams.at / V3_DEFAULTS.at));
  // virtual clock: advances at wall-clock × speed so the slider acts live
  let vt = 0;
  let lastNow = performance.now();
  const el = $('plot-poly');
  const scratch = new Float32Array(t.NI * 3);
  const trails = Array.from({ length: t.NI }, () => ({
    x: [] as number[], y: [] as number[], z: [] as number[],
  }));
  // global kmax from ilv 0, rep 0 — magnitude law is shared across ilvs
  let kmax = 1e-9;
  for (let p = 0; p < t.NPTS; p++) kmax = Math.max(kmax, Math.hypot(t.kx[p], t.ky[p], t.kz[p]));
  // camera: fixed distance, user-driven via the zoom slider (live each frame)
  const CAM_FAR = 2.17;                   // plotly default eye distance = 1x
  const step = () => {
    const now = performance.now();
    vt += (now - lastNow) * polySpeed();
    lastNow = now;
    const done = vt >= REP_MS;             // single readout; last frame = full k
    const ph = done ? 1 : vt / REP_MS;
    const p = Math.min(t.NPTS - 1, Math.floor(ph * (t.NPTS - 1)));
    const TRAIL = polyTrail();
    let scale = 0;
    for (let j = 0; j < t.NI; j++) {
      const lin = p + j * t.NPTS;          // rep 0 only
      const x = t.kx[lin] / kmax, y = t.ky[lin] / kmax, z = t.kz[lin] / kmax;
      scratch[3 * j] = x; scratch[3 * j + 1] = y; scratch[3 * j + 2] = z;
      scale = Math.max(scale, Math.hypot(x, y, z));
      const tr = trails[j];
      tr.x.push(x); tr.y.push(y); tr.z.push(z);
      while (tr.x.length > TRAIL) { tr.x.shift(); tr.y.shift(); tr.z.shift(); }
    }
    plots.plotPolyhedron(el, scratch, t.NI,
      'readout animation — one interleaf set, rep 0', trails, CAM_FAR / polyZoom());
    $('poly-frame').textContent =
      `t ${(ph * lastParams.at * 1e3).toFixed(2)} ms · r/kmax ${scale.toFixed(2)}`;
    if (done) {
      stopPolyAnim();
      $('poly-frame').textContent = 'done — one readout, resting at kmax';
      return;
    }
    polyAnimId = requestAnimationFrame(step);
  };
  $('poly-animate').textContent = '⏸ stop';
  polyAnimId = requestAnimationFrame(step);
}

function renderPolyhedron() {
  if (!traj) return;
  stopPolyAnim();
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
    ['Nyquist radius r_N (grid units)', r => r.rN !== undefined ? `${fmt(r.rN, 0)} of kmax ${fmt(r.kmaxGrid!, 0)}` : '—'],
    ['r_N full set', r => r.rNFull !== undefined ? fmt(r.rNFull, 0) : '—'],
  ];
  let h = '<table class="metrics"><tr><th>metric</th>' + rs.map((_, i) => `<th>#${i + 1}</th>`).join('') + '</tr>';
  for (const [name, get, headline] of rows)
    h += `<tr><td class="name">${name}</td>` +
         rs.map(r => `<td class="${headline ? 'headline' : ''}">${get(r)}</td>`).join('') + '</tr>';
  h += '</table><div class="note">max/noise-like = 1.0 → perfectly incoherent (noise-like) aliasing; ' +
       'v3 aligned windows measure ≈ 1.5. σ is Rayleigh-correct: √(E|x|²/2) over r &gt; 2·FWHM. ' +
       'r_N = radius where shell sample density drops below 1 per (1/FOV)³ cell — ' +
       'Nyquist-complete inside, undersampled outside (see Projections tab for the density curve).</div>';
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
  let n = parseInt(($('p-psfn') as HTMLInputElement).value, 10) || 64;
  n = Math.min(128, Math.max(16, n - (n % 2)));
  ($('p-psfn') as HTMLInputElement).value = `${n}`;
  const label = ($('ens-expr') as HTMLInputElement).value.trim();
  ($('btn-psf') as HTMLButtonElement).disabled = true;
  try {
    lastPsf = await request<PsfResult>('psf', { ilvs, n, label });
    if (traj) {
      const ensD = shellDensity(traj, ilvs), fullD = shellDensity(traj, allIlvs());
      lastPsf.rN = ensD.rN; lastPsf.rNFull = fullD.rN; lastPsf.kmaxGrid = ensD.kmaxGrid;
    }
    const slider = $('slice-idx') as HTMLInputElement;
    slider.max = `${lastPsf.n - 1}`;
    slider.value = `${lastPsf.n >> 1}`;
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
