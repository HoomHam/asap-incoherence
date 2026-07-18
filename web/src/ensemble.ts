// Ensemble picker expression parser.
// Global interleave index g = irep*NI + j, g in [0, NI*NREPS).
// Grammar (comma-separated terms, union):
//   all              -> every interleave
//   12               -> single interleave
//   0-25             -> inclusive range
//   block:k / rot:k  -> aligned set k = [k*NI, (k+1)*NI)
//   off:o            -> misaligned NI-window starting at o = [o, o+NI)
//   every:s          -> every s-th interleave (0, s, 2s, ...)
//   every:s:from     -> every s-th starting at `from`

export function parseEnsemble(expr: string, NI: number, NREPS: number): number[] {
  const total = NI * NREPS;
  const set = new Set<number>();
  const add = (g: number) => { if (g >= 0 && g < total) set.add(g); };
  for (const raw of expr.split(',')) {
    const term = raw.trim().toLowerCase();
    if (!term) continue;
    if (term === 'all') {
      for (let g = 0; g < total; g++) add(g);
    } else if (term.startsWith('block:') || term.startsWith('rot:')) {
      const k = parseInt(term.split(':')[1], 10);
      if (Number.isNaN(k)) throw new Error(`bad term "${term}"`);
      for (let j = 0; j < NI; j++) add(k * NI + j);
    } else if (term.startsWith('off:')) {
      const o = parseInt(term.split(':')[1], 10);
      if (Number.isNaN(o)) throw new Error(`bad term "${term}"`);
      for (let j = 0; j < NI; j++) add(o + j);
    } else if (term.startsWith('every:')) {
      const parts = term.split(':').slice(1).map(s => parseInt(s, 10));
      const s = parts[0], from = parts[1] ?? 0;
      if (!s || s < 1) throw new Error(`bad term "${term}"`);
      for (let g = from; g < total; g += s) add(g);
    } else if (term.includes('-')) {
      const [a, b] = term.split('-').map(s => parseInt(s, 10));
      if (Number.isNaN(a) || Number.isNaN(b)) throw new Error(`bad term "${term}"`);
      for (let g = a; g <= b; g++) add(g);
    } else {
      const g = parseInt(term, 10);
      if (Number.isNaN(g)) throw new Error(`bad term "${term}"`);
      add(g);
    }
  }
  return [...set].sort((a, b) => a - b);
}

export const PRESETS: { label: string; expr: (NI: number, NREPS: number) => string }[] = [
  { label: '1 interleave', expr: () => '0' },
  { label: '1 block (aligned set)', expr: () => 'block:0' },
  { label: 'misaligned window (off 13)', expr: () => 'off:13' },
  { label: '4 blocks', expr: NI => `0-${4 * NI - 1}` },
  { label: 'every 4th', expr: () => 'every:4' },
  { label: 'full set', expr: () => 'all' },
];
