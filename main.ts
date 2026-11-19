// gta6-trophy-rarity-forecast-01 entry (Node 22, standard library only).
//
// Forecast: for each trophy, the earn rate on Exophase at the case's target.
// Model, in earners rather than rates. Let P be the players tracked and
// E_i the earners of trophy i, so the rate is r_i = 100 * E_i / P.
//
// Exophase refreshes its counts on its own schedule, not continuously: the
// page can sit unchanged for hours and then jump. A refresh does two things
// at once: new players join (P jumps, diluting every rate) and the players
// already tracked have earned more trophies (E_i grows). The net rate move
// is captured by one elasticity per trophy, beta_i:
//
//   E_i grows as P^beta_i, so r_i* = r_i * (P* / P)^(beta_i - 1)
//
// beta_i > 1 means the trophy's rate rises as the player base grows (early,
// easy trophies players reach quickly); beta_i < 1 means it falls (rare or
// late trophies the flood of new players does not hold). beta_i is fitted
// from the case's own refreshes (pooled slope of log E_i on log P over the
// snapshots), shrunk toward a prior that depends on the trophy's current
// rate and tier, with the shrinkage weight set by how many earners the
// trophy has. When the history shows no refresh at all (launch night: the
// page flat since the first read), there is nothing to fit, so the program
// falls back to launch-night priors: a refresh is more likely than not
// before the target, it multiplies the tracked players several-fold, and
// earners grow only slightly on it (new players hold almost no trophies
// yet), so every rate but the 100% one falls hard. The forecast is a
// two-scenario mixture -- no refresh (persistence) vs refresh (the growth
// model) -- and the five quantiles are the mixture's quantiles on the logit
// scale, so the band honestly spans both outcomes.
//
// The 100% trophy stays 100%: every tracked player holds it, new players
// included, and the rules say the refresh left it alone.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Random-walk spread of a rate's move, in logit units per square-root day. */
const SIGMA_RW = 0.5;
/** Launch-night prior: how many times the tracked players multiply at a refresh. */
const GROWTH_PRIOR = 4;
/** Launch-night prior: earner elasticity when no refresh has been seen. */
const BETA_FLAT = 0.15;
/** Launch-night prior: probability a refresh lands before the target. */
const W_REFRESH_PRIOR = 0.65;
/** Extra logit spread for the growth scenario when it rests on priors. */
const SPREAD_GROWTH_PRIOR = 0.45;
/** Extra logit spread for the growth scenario when its trend is observed. */
const SPREAD_GROWTH_DATA = 0.2;
/** Small-count spread: one earner more or less moves a rare trophy a lot. */
const SPREAD_COUNT = 0.5;

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  game: { release_date: string };
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clamp(pct);
  return Math.log(p / (100 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** Standard normal CDF. */
const normCdf = (x: number): number => 0.5 * (1 + erf(x / Math.SQRT2));
// Error function, Abramowitz and Stegun 7.1.26 (|eps| < 1.5e-7).
const erf = (x: number): number => {
  const s = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
};
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Collapse the history into distinct snapshots: a read that changed nothing. */
const snapshots = (history: Read[]): Read[] => {
  const snaps: Read[] = [];
  for (const h of history) {
    const prev = snaps[snaps.length - 1];
    if (prev !== undefined && h.players_tracked === prev.players_tracked && h.rates.every((r, i) => r === prev.rates[i])) continue;
    snaps.push(h);
  }
  return snaps;
};

/** Least-squares slope and intercept of y on x. */
const linFit = (xs: number[], ys: number[]): { a: number; b: number } => {
  const n = xs.length;
  const mx = xs.reduce((s, x) => s + x, 0) / n;
  const my = ys.reduce((s, y) => s + y, 0) / n;
  let den = 0;
  let num = 0;
  for (let i = 0; i < n; i++) {
    den += (xs[i] - mx) * (xs[i] - mx);
    num += (xs[i] - mx) * (ys[i] - my);
  }
  const b = den > 0 ? num / den : 0;
  return { a: my - b * mx, b };
};

/** Quantiles of a mixture of logit-normals, by bisection on the mixture CDF. */
const mixtureQuantiles = (components: Array<{ w: number; mu: number; s: number }>): number[] => {
  const out: number[] = [];
  for (const tau of LEVELS) {
    let lo = -14;
    let hi = 12;
    for (let k = 0; k < 80; k++) {
      const mid = 0.5 * (lo + hi);
      let cdf = 0;
      for (const c of components) cdf += c.w * normCdf((mid - c.mu) / c.s);
      if (cdf < tau) lo = mid;
      else hi = mid;
    }
    out.push(invLogit(0.5 * (lo + hi)));
  }
  return out;
};

const forecastCase = (c: Case): Array<{ id: number; q: number[] }> => {
  const snaps = snapshots(c.history);
  const last = snaps[snaps.length - 1];
  if (last === undefined) throw new Error('case has no history');
  const pNow = last.players_tracked;
  const horizonDays = c.horizon_minutes / 1440;
  const release = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const tTarget = Date.parse(c.target);
  const sRw = SIGMA_RW * Math.sqrt(horizonDays);

  // --- players at the target, and the chance a refresh lands before it ---
  let pStar: number;
  let wRefresh: number;
  let betaHat: Array<number | null>;
  if (snaps.length >= 2) {
    // Players grow roughly as a power of time since release; fit log P on
    // log(1 + hours) over the snapshots and read the line at the target.
    const xs = snaps.map((s) => Math.log1p(Math.max((Date.parse(s.at) - release) / 3_600_000, 0)));
    const ys = snaps.map((s) => Math.log(s.players_tracked));
    const fit = linFit(xs, ys);
    pStar = Math.exp(fit.a + fit.b * Math.log1p(Math.max((tTarget - release) / 3_600_000, 0)));
    pStar = Math.min(Math.max(pStar, pNow), pNow * 50);
    // The refresh cadence is the median gap between the snapshots; the
    // chance at least one refresh lands before the target follows from it.
    // (A horizon far longer than the cadence makes a refresh near-certain.)
    const gaps: number[] = [];
    for (let j = 1; j < snaps.length; j++) gaps.push((Date.parse(snaps[j].at) - Date.parse(snaps[j - 1].at)) / 3_600_000);
    gaps.sort((a, b) => a - b);
    const cadenceHours = Math.max(gaps[Math.floor(gaps.length / 2)], 0.5);
    wRefresh = Math.min(0.999, Math.max(0.5, 1 - Math.exp(-(c.horizon_minutes / 60) / cadenceHours)));
    // Earner elasticity per trophy: pooled slope of log E on log P.
    betaHat = c.trophies.map((_t, i) => {
      const ex: number[] = [];
      const ey: number[] = [];
      for (const s of snaps) {
        const r = s.rates[i];
        const e = r === null ? 0 : (r * s.players_tracked) / 100;
        if (e > 0) {
          ex.push(Math.log(s.players_tracked));
          ey.push(Math.log(e));
        }
      }
      if (ex.length < 2) return null;
      return linFit(ex, ey).b;
    });
  } else {
    // Flat history: no refresh has been seen, so fall back to launch-night
    // priors -- a refresh is more likely than not, and it multiplies the
    // tracked players several-fold.
    pStar = pNow * GROWTH_PRIOR;
    wRefresh = W_REFRESH_PRIOR;
    betaHat = c.trophies.map(() => null);
  }

  return c.trophies.map((t, i) => {
    const r = last.rates[i] ?? 0.01;
    // A trophy every tracked player holds stays held: the refresh left the
    // 100% rate alone, and new players hold it too.
    if (r >= 99.99) return { id: t.id, q: [99.99, 99.99, 99.99, 100, 100] };
    const eNow = (r * pNow) / 100;
    // Prior elasticity: early/common trophies keep up with player growth,
    // rare ones do not; the platinum needs every other trophy, so it grows
    // slowest of all.
    let betaPrior = Math.min(1, Math.max(0.1, 0.25 + 0.35 * Math.log10(Math.max(r, 0.01) / 0.1)));
    if (t.tier === 'platinum') betaPrior = Math.min(betaPrior, 0.4);
    let beta: number;
    if (betaHat[i] === null) {
      beta = snaps.length < 2 ? BETA_FLAT : betaPrior;
    } else {
      // Trust the fitted slope more the more earners the trophy has.
      const wData = eNow / (eNow + 30);
      beta = wData * Math.min(Math.max(betaHat[i] as number, 0), 2) + (1 - wData) * betaPrior;
    }
    beta = Math.min(Math.max(beta, 0), 1.5);
    // The growth scenario's centre: earners grow as P^beta.
    const rStar = clamp(r * Math.pow(pStar / pNow, beta - 1));
    const sCount = SPREAD_COUNT / Math.sqrt(eNow + 1);
    const sGrowth = snaps.length < 2 ? SPREAD_GROWTH_PRIOR : SPREAD_GROWTH_DATA;
    const sB = Math.sqrt(sRw * sRw + sGrowth * sGrowth + sCount * sCount);
    const sA = Math.sqrt(sRw * sRw + sCount * sCount);
    const q = mixtureQuantiles([
      { w: 1 - wRefresh, mu: logit(r), s: Math.max(sA, 0.05) },
      { w: wRefresh, mu: logit(rStar), s: Math.max(sB, 0.05) },
    ]);
    const rounded = q.map((v) => round4(Math.min(100, Math.max(0, v))));
    for (let j = 1; j < rounded.length; j++) rounded[j] = Math.max(rounded[j], rounded[j - 1]);
    return { id: t.id, q: rounded };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
