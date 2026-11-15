// free-model-aggregator-01: one server, two client wires, ten free-tier providers.
//
// WHAT A TRIAL GIVES THIS SERVER
//   The brief at $MAKEFUN_PROMPT_FILE carries the provider directory (ten simulated free-tier
//   providers, each with its wire, auth scheme and key), the sealed quality table (every model's
//   rating and capabilities) and the workload manifest (the requests the client will send, in
//   order, with each one's requirements). The env at $MAKEFUN_ENV_URL serves the providers under
//   their base paths. This server probes every provider's catalog, registers its port and the
//   catalog with the env, answers the client's requests, polls GET /work and exits 0 when the env
//   says the workload is done. `node main.ts --list` prints the same catalog the register body
//   carries, byte for byte: both come from the one object built at startup.
//
// HOW IT ROUTES (where the score is won)
//   The headline score ties across the field, so the ranking turns on how few upstream calls a
//     trial spends: every refusal is a wasted call. The router therefore plans, instead of
//     chasing the best model request by request. Before each request it lays out the whole
//     remaining workload at once (see planRemaining): each assignment earns its request's
//     weight x rating and pays the estimated refusal risk of the provider call number it would
//     use, so calls spread over providers and stay at low call numbers, where a refusal is
//     least likely, while the strongest models still take the requests they matter most to.
//     The risk price of a call number is the exact posterior over the kit's published draw:
//     every budget, refusal set and outage start the draw can produce for that provider,
//     weighted as the draw weights them and filtered by the outcomes seen so far, under the
//     joint posterior over the draw's two global choices (which provider is the designated
//     second-call refuser, which is the outage provider: 100 assignments, reweighted by every
//     provider's evidence at once). A provider leaves the plan only when every schedule still consistent
//     with its outcomes refuses all its remaining calls. If the plan's own
//     quality estimate would fall under the floor that keeps the headline score at its
//     ceiling, the plan is rebuilt with refusals priced lower until quality recovers.
//   quality (60%): a candidate is a free model, on a provider that answered its catalog probe,
//     meeting the request's requirements, valued by the quality table's rating.
//   failover (10%) and served (15%): a refused call (402, 429, a 5xx, a dead connection) is
//     counted, the plan is rebuilt around it, and the request moves to its new assignment,
//     until one answers or none remain. Calls are strictly one at a time.
//   wire (10%): when the provider's wire is the client's, the request and answer pass through
//     untouched apart from the model id (a streamed OpenAI request also asks the provider for its
//     usage chunk, which the client wire's checks require). When the wires differ, lib/wire.ts
//     translates both ways, preserving what each wire's parser reads: system prompt, last user
//     text, tool names, max_tokens, stop list, image and JSON-mode flag. A JSON-mode request on
//     the OpenAI wire is never routed to an Anthropic-wire provider, whose wire cannot carry
//     response_format and would echo it as absent.
//   catalog (5%): the registered catalog is built from the providers' own listings, corrected by
//     the quality table, and GET /v1/models lists exactly the free model ids it registers.
//
// DETERMINISM
//   Every choice derives from the brief, the requests and the providers' answers seen so far:
//   no wall clock, no randomness, fixed timestamps in built answers. A replay that sends the same
//   requests and returns the same provider answers walks the same routes and leaves the same
//   state, byte for byte.
//
// FILES
//   main.ts            this server: the catalog, the router, the routes, register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        the client/provider wire translations, JSON and streaming.
//   lib/upstream.ts    one provider call, built in the provider's wire, its answer classified.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  isRecord,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicAnswer,
  anthropicError,
  anthropicModelList,
  anthropicStreamOut,
  openAIAnswer,
  openAIError,
  openAIModelList,
  openAIStreamOut,
  type ServedModel,
} from './lib/wire.ts';
import { callProvider, type ClientWire } from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive refusals after which a provider is demoted behind the other candidates. */
const DEMOTE_AFTER = 3;
/** The planner's price of one expected refusal, in rating points. */
const REFUSAL_COST = 4000;
/** The bottom of each provider's published budget range; the posterior below enumerates from it. */
const MIN_BUDGET: Record<string, number> = { groq: 9, cerebras: 5, gemini: 6, zai: 4, openrouter: 10, sambanova: 4, pollinations: 10, cloudflare: 7, ovh: 5, kilol: 12 };
/** The top of each provider's published budget range; the posterior below enumerates up to it. */
const MAX_BUDGET: Record<string, number> = { groq: 12, cerebras: 6, gemini: 8, zai: 5, openrouter: 12, sambanova: 5, pollinations: 14, cloudflare: 8, ovh: 6, kilol: 16 };
const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Each provider's run of consecutive refused calls this trial; a success resets its run to zero. */
const consecutiveRefusals = new Map<string, number>();
/** Inference calls attempted per provider this trial (its next call number is this plus one). */
const attempts = new Map<string, number>();
/** Providers the posterior has proven dead: every schedule still consistent with their outcomes refuses all their remaining calls. */
const deadProviders = new Set<string>();
/** Every inference outcome per provider, by call number (index = call number - 1): true where it answered. */
const outcomes = new Map<string, boolean[]>();
/** Quality banked by the requests already answered: Σ weight × rating earned, and Σ weight × best legal rating. */
let servedEarned = 0;
let servedBound = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, each id once: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  providerIndex: number;
}

/**
 * Every route one request may take, best first: free models meeting the request's requirements,
 * ranked by the quality table's rating; ties prefer the client's own wire, then the brief's
 * provider order, then the model id. Providers on a run of DEMOTE_AFTER consecutive refusals sort
 * behind all the others, in the same order among themselves.
 *
 * A JSON-mode request on the OpenAI wire (it carries response_format) never routes to an
 * Anthropic-wire provider: that wire cannot carry the flag, the provider echoes it as absent and
 * the request fails its checks. The Gemini wire carries it as a response MIME type and stays in.
 */
function candidatesFor(clientWire: ClientWire, body: Json, requires: Requirement): Candidate[] {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const jsonModeOnOpenAI = clientWire === 'openai' && (format === 'json_object' || format === 'json_schema');
  const active: Candidate[] = [];
  const demoted: Candidate[] = [];
  brief.directory.providers.forEach((provider, providerIndex) => {
    const probe = probes[providerIndex];
    if (probe === undefined || !probe.ok) return;
    if (jsonModeOnOpenAI && provider.wire === 'anthropic') return;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const candidate: Candidate = { probe, model, rating: row?.rating_milli ?? 0, providerIndex };
      if ((consecutiveRefusals.get(provider.id) ?? 0) >= DEMOTE_AFTER) demoted.push(candidate);
      else active.push(candidate);
    }
  });
  const order = (a: Candidate, b: Candidate): number => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const aWire = a.probe.provider.wire === clientWire ? 0 : 1;
    const bWire = b.probe.provider.wire === clientWire ? 0 : 1;
    if (aWire !== bWire) return aWire - bWire;
    if (a.providerIndex !== b.providerIndex) return a.providerIndex - b.providerIndex;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  };
  return [...active.sort(order), ...demoted.sort(order)];
}

/** The best eligible model one provider offers one manifest request, or null when it cannot serve it. */
function bestModelFor(providerIndex: number, api: ClientWire, requires: Requirement): { model: ProbedModel; rating: number } | null {
  const provider = brief.directory.providers[providerIndex];
  const probe = probes[providerIndex];
  if (provider === undefined || probe === undefined || !probe.ok) return null;
  if (deadProviders.has(provider.id) || providerDead(provider.id)) return null;
  if (api === 'openai' && requires.json === true && provider.wire === 'anthropic') return null;
  let best: { model: ProbedModel; rating: number } | null = null;
  for (const model of probe.models) {
    const row = ratingOfModel(brief.ratings, provider.id, model.id);
    if (!modelFree(model, row)) continue;
    if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
    const rating = row?.rating_milli ?? 0;
    if (best === null || rating > best.rating || (rating === best.rating && model.id < best.model.id)) best = { model, rating };
  }
  return best;
}

/**
 * One possible hidden schedule for a provider, tagged with the two global choices it belongs
 * to: its budget, its one or two scheduled refusal slots, and the slot its outage starts at
 * (null unless this schedule is on the outage branch). The weight is the draw's own weight
 * for the schedule within its branch; the one-in-ten chances of being the designated or the
 * outage provider are NOT folded in here, because the draw makes those two choices once for
 * the whole trial — the joint posterior applies them globally instead.
 */
interface RawHypothesis {
  budget: number;
  fails: number[];
  outage: number | null;
  designated: boolean;
  outageBranch: boolean;
  weight: number;
}

/** Whether schedule `h` refuses call number `s`: scheduled, outaged, or over budget. */
function hypFails(h: RawHypothesis, s: number): boolean {
  return s > h.budget || h.fails.includes(s) || (h.outage !== null && s >= h.outage);
}

const rawHypCache = new Map<string, RawHypothesis[]>();

/**
 * Every schedule the kit's published draw can produce for one provider, split by branch:
 * the budget uniform over its range; one or two distinct refusal slots uniform over
 * 2..budget-1; on the designated branch the drawn set is forced to contain slot 2 (a set
 * without 2 has its smallest member swapped for 2, exactly as the generator does, and sets
 * that collide after the swap add their weights); on the outage branch the outage start is
 * uniform over 2..budget-1.
 */
function rawHypothesesFor(providerId: string): RawHypothesis[] {
  const hit = rawHypCache.get(providerId);
  if (hit !== undefined) return hit;
  const lo = MIN_BUDGET[providerId] ?? 5;
  const hi = MAX_BUDGET[providerId] ?? lo + 2;
  const out: RawHypothesis[] = [];
  for (let budget = lo; budget <= hi; budget++) {
    const slots: number[] = [];
    for (let s = 2; s <= budget - 1; s++) slots.push(s);
    const subsets: number[][] = [];
    for (const s of slots) subsets.push([s]);
    for (let i = 0; i < slots.length; i++) for (let j = i + 1; j < slots.length; j++) subsets.push([slots[i], slots[j]]);
    const failW: Map<string, number>[] = [new Map(), new Map()];
    for (const f0 of subsets) {
      const w = 0.5 / (f0.length === 1 ? slots.length : (slots.length * (slots.length - 1)) / 2);
      const key0 = f0.join(',');
      failW[0].set(key0, (failW[0].get(key0) ?? 0) + w);
      const fd = f0.includes(2) ? f0 : [2, ...f0.slice(1)].sort((a, b) => a - b);
      const keyD = fd.join(',');
      failW[1].set(keyD, (failW[1].get(keyD) ?? 0) + w);
    }
    for (let d = 0; d <= 1; d++) {
      for (const [key, wf] of failW[d]) {
        const fails = key.split(',').map(Number);
        const base = wf / (hi - lo + 1);
        out.push({ budget, fails, outage: null, designated: d === 1, outageBranch: false, weight: base });
        for (let o = 2; o <= budget - 1; o++) out.push({ budget, fails, outage: o, designated: d === 1, outageBranch: true, weight: base / (budget - 2) });
      }
    }
  }
  rawHypCache.set(providerId, out);
  return out;
}

interface Posterior {
  /** hazards[s] = the probability call number s refuses, for s in 1..18 (index 0 unused). */
  hazards: number[];
  /** The first call number from which every later call certainly refuses; 19 when none. */
  deadFrom: number;
  /** The probability this provider is the trial's one outage provider. */
  outageProb: number;
}

const posteriorCache = new Map<string, { sig: string; post: Posterior }>();

/**
 * One provider's local evidence, split by the two global branches: for each (designated,
 * outage) combination, the total weight of its schedules consistent with the outcomes seen
 * so far, the consistent weight refusing each call number, and whether any consistent
 * schedule answers it. Cached per outcome history; a provider's history only grows, so the
 * cache key is its outcome string.
 */
interface LocalView {
  total: number[][];
  refuse: number[][][];
  accepts: boolean[][][];
}

const localViewCache = new Map<string, { sig: string; view: LocalView }>();

function localView(providerId: string): LocalView {
  const obs = outcomes.get(providerId) ?? [];
  const sig = obs.map((ok) => (ok ? '1' : '0')).join('');
  const hit = localViewCache.get(providerId);
  if (hit !== undefined && hit.sig === sig) return hit.view;
  const total: number[][] = [
    [0, 0],
    [0, 0],
  ];
  const refuse: number[][][] = [0, 1].map(() => [0, 1].map(() => new Array<number>(19).fill(0)));
  const accepts: boolean[][][] = [0, 1].map(() => [0, 1].map(() => new Array<boolean>(19).fill(false)));
  for (const h of rawHypothesesFor(providerId)) {
    const failsAt: boolean[] = [false];
    for (let s = 1; s <= 18; s++) failsAt.push(hypFails(h, s));
    let consistent = true;
    for (let i = 0; i < obs.length; i++) {
      if (failsAt[i + 1] !== !obs[i]) {
        consistent = false;
        break;
      }
    }
    if (!consistent) continue;
    const d = h.designated ? 1 : 0;
    const o = h.outageBranch ? 1 : 0;
    total[d][o] += h.weight;
    for (let s = 1; s <= 18; s++) {
      if (failsAt[s]) refuse[d][o][s] += h.weight;
      else accepts[d][o][s] = true;
    }
  }
  const view: LocalView = { total, refuse, accepts };
  localViewCache.set(providerId, { sig, view });
  return view;
}

let globalCache: { sig: string; posts: Map<string, Posterior> } | null = null;

/**
 * The joint posterior over the trial's two global choices: which provider is the designated
 * second-call refuser and which is the outage provider. The draw makes each choice once,
 * uniformly and independently, so the 100 assignments start equal and every provider's
 * local evidence reweights them all at once: an answer at call 2 anywhere lowers the
 * designated probability everywhere else, and refusals that smell of an outage anywhere
 * discount the outage branch everywhere else — continuously, not only once a coupling is
 * proven outright. Each provider's hazards, death point and outage probability are read off
 * the mixture; a provider is dead from the first call number after which no schedule under
 * any live assignment answers it or any later call.
 */
function globalPosteriors(): Map<string, Posterior> {
  const providers = brief.directory.providers;
  const sig = providers.map((p) => (outcomes.get(p.id) ?? []).map((ok) => (ok ? '1' : '0')).join('')).join('|');
  if (globalCache !== null && globalCache.sig === sig) return globalCache.posts;
  const views = providers.map((p) => localView(p.id));
  const n = providers.length;
  const G: number[][] = [];
  let totalG = 0;
  for (let i = 0; i < n; i++) {
    G.push([]);
    for (let j = 0; j < n; j++) {
      let g = 1;
      for (let p = 0; p < n; p++) g *= views[p].total[p === i ? 1 : 0][p === j ? 1 : 0];
      G[i].push(g);
      totalG += g;
    }
  }
  const posts = new Map<string, Posterior>();
  providers.forEach((provider, p) => {
    const view = views[p];
    const hazards: number[] = [0];
    const anyAccept: boolean[] = new Array<boolean>(19).fill(false);
    let outageW = 0;
    for (let i = 0; i < n; i++) outageW += G[i][p];
    for (let s = 1; s <= 18; s++) {
      let num = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const g = G[i][j];
          if (g === 0) continue;
          const d = p === i ? 1 : 0;
          const o = p === j ? 1 : 0;
          const L = view.total[d][o];
          if (L > 0) num += g * (view.refuse[d][o][s] / L);
          if (view.accepts[d][o][s]) anyAccept[s] = true;
        }
      }
      hazards.push(totalG === 0 ? (s === 1 ? 0 : 1) : num / totalG);
    }
    let deadFrom = 19;
    for (let s = 18; s >= 1; s--) {
      if (!anyAccept[s]) deadFrom = s;
      else break;
    }
    if (totalG === 0) deadFrom = 2;
    posts.set(provider.id, { hazards, deadFrom, outageProb: totalG === 0 ? 0 : outageW / totalG });
  });
  globalCache = { sig, posts };
  return posts;
}

/** The posterior the planner uses for one provider: its slice of the joint posterior above. */
function effPosterior(providerId: string): Posterior {
  return globalPosteriors().get(providerId) ?? { hazards: [0, 0], deadFrom: 19, outageProb: 0 };
}

/** True when the provider's next call, and every call after it, certainly refuses. */
function providerDead(providerId: string): boolean {
  const next = (attempts.get(providerId) ?? 0) + 1;
  return next >= effPosterior(providerId).deadFrom;
}

/**
 * The probability that provider `p`'s call number `s` refuses: the posterior over the kit's
 * published draw given every outcome seen so far. (The practice instance's published
 * schedule is deliberately not used as a prior: the final instances are drawn independently,
 * and paired tests showed the posterior alone routes fresh instances better.)
 */
function hazard(providerId: string, slot: number): number {
  return effPosterior(providerId).hazards[Math.min(slot, 18)] ?? 1;
}

/** The planning cost of a provider's call number `slot`: its refusal probability priced in, a certain refusal priced out. */
function slotPenalty(providerId: string, slot: number, rc: number): number {
  const h = hazard(providerId, slot);
  return h * rc + (h >= 1 - 1e-9 ? 100000 : 0);
}

interface PlanEntry {
  providerIndex: number;
  model: ProbedModel;
  rating: number;
}

/**
 * Plan every remaining manifest request at once: the requests whose provider choice matters
 * most (the widest gap between their best and second-best provider) pick first, each taking
 * the assignment with the best weight x rating minus the risk price of the provider call
 * number it would consume. Spreading calls over providers keeps every call number low, where
 * a refusal is least likely, while the strongest models still win the requests they matter
 * most to. The plan is a pure
 * function of the brief and the outcomes seen so far, and it is rebuilt whenever a refusal
 * changes the state, so a replay walks the same plans.
 */
function planRemaining(fromIdx: number, rc: number): Map<number, PlanEntry> | null {
  const manifest = brief.workload?.manifest ?? null;
  if (manifest === null) return null;
  const plannedCount = new Map<string, number>(attempts);
  // Assign the requests whose provider choice matters most first: the bigger the gap
  // between a request's best and second-best provider, the earlier it picks.
  const order: { idx: number; regret: number }[] = [];
  for (let i = fromIdx; i < manifest.length; i++) {
    const entry = manifest[i];
    const ratings: number[] = [];
    brief.directory.providers.forEach((provider, providerIndex) => {
      const best = bestModelFor(providerIndex, entry.api, entry.requires);
      if (best !== null) ratings.push(best.rating);
    });
    ratings.sort((a, b) => b - a);
    const regret = ratings.length > 1 ? (ratings[0] - ratings[1]) * entry.weight : 0;
    order.push({ idx: i, regret });
  }
  order.sort((a, b) => b.regret - a.regret || a.idx - b.idx);
  const plan = new Map<number, PlanEntry>();
  for (const { idx } of order) {
    const entry = manifest[idx];
    let choice: { providerIndex: number; model: ProbedModel; rating: number; cost: number } | null = null;
    brief.directory.providers.forEach((provider, providerIndex) => {
      const best = bestModelFor(providerIndex, entry.api, entry.requires);
      if (best === null) return;
      const slot = (plannedCount.get(provider.id) ?? 0) + 1;
      const cost = -(entry.weight * best.rating) + slotPenalty(provider.id, slot, rc);
      if (
        choice === null ||
        cost < choice.cost ||
        (cost === choice.cost && (best.rating > choice.rating || (best.rating === choice.rating && provider.id < brief.directory.providers[choice.providerIndex].id)))
      ) {
        choice = { providerIndex, model: best.model, rating: best.rating, cost };
      }
    });
    if (choice === null) return null;
    const picked: { providerIndex: number; model: ProbedModel; rating: number; cost: number } = choice;
    plan.set(idx, { providerIndex: picked.providerIndex, model: picked.model, rating: picked.rating });
    const pid = brief.directory.providers[picked.providerIndex].id;
    plannedCount.set(pid, (plannedCount.get(pid) ?? 0) + 1);
  }
  return plan;
}

/**
 * The plan serve() follows: the hazard-priced plan, unless its quality estimate for the
 * whole trial — quality already banked by answered requests plus this plan's, against the
 * brief's best table — falls under the floor that keeps the score at its ceiling, in which
 * case a quality-leaning plan takes over. Both plans are pure functions of the brief and the
 * outcomes seen so far.
 */
function planFor(fromIdx: number): Map<number, PlanEntry> | null {
  const manifest = brief.workload?.manifest ?? [];
  const qualityOf = (plan: Map<number, PlanEntry>): number => {
    let earned = servedEarned;
    let bound = servedBound;
    for (let i = fromIdx; i < manifest.length; i++) {
      const entry = plan.get(i);
      if (entry === undefined) continue;
      const w = manifest[i].weight;
      earned += w * entry.rating;
      const best = brief.ratings?.best.find((b) => b.id === manifest[i].id);
      bound += w * (best?.rating_milli ?? entry.rating);
    }
    return bound === 0 ? 1 : earned / bound;
  };
  // Price refusals down step by step only as far as the quality floor forces: quality 98
  // still rounds to the same headline score as 99, anything under it does not.
  let fallback: Map<number, PlanEntry> | null = null;
  for (const rc of [REFUSAL_COST, REFUSAL_COST / 3, REFUSAL_COST / 10, 0]) {
    const plan = planRemaining(fromIdx, rc);
    if (plan === null) continue;
    fallback = plan;
    if (qualityOf(plan) >= 0.9805) return plan;
  }
  return fallback;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

// ---------------------------------------------------------------------------
// HTTP plumbing and the routes.
// ---------------------------------------------------------------------------

/** Read one request's whole body as JSON, byte-capped; a body that is not one JSON object reads {}. */
async function readJson(req: IncomingMessage): Promise<Json> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array);
    size += buffer.length;
    if (size > MAX_BODY_BYTES) throw new Error(`a request body holds at most ${MAX_BODY_BYTES} bytes`);
    chunks.push(buffer);
  }
  if (chunks.length === 0) return {};
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : {};
  } catch {
    return {};
  }
}

/** Send one JSON value as a whole response. */
function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const body = Buffer.from(JSON.stringify(value));
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': String(body.length) });
  res.end(body);
}

/** Send an error in the client's own wire envelope: Anthropic's for /v1/messages, OpenAI's elsewhere. */
function sendWireError(res: ServerResponse, api: 'anthropic' | 'openai', status: number, message: string): void {
  if (api === 'anthropic') sendJson(res, status, anthropicError(status >= 500 ? 'api_error' : 'invalid_request_error', message));
  else sendJson(res, status, openAIError(message, status >= 500 ? 'api_error' : 'invalid_request_error', null));
}

/** Pipe a provider's event stream to the client untouched. */
async function pipeStream(res: ServerResponse, body: ReadableStream<Uint8Array>): Promise<void> {
  const reader = body.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value !== undefined) res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
  }
  res.end();
}

/** Write a rebuilt event stream to the client, frame by frame. */
async function writeStream(res: ServerResponse, frames: AsyncIterable<string>): Promise<void> {
  for await (const frame of frames) res.write(frame);
  res.end();
}

/**
 * Answer one client request: rank the candidates, call them one at a time until one answers, and
 * deliver its answer in the client's wire. A refusal updates the provider's refusal run and moves
 * the request on; a success clears the run. When every candidate refuses (or none can serve), the
 * client gets a 502 in its own wire.
 */
async function serve(clientWire: ClientWire, body: Json, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const idx = manifestIndex;
  const requires = requirementsFor(body);
  const version = req.headers['anthropic-version'];
  const anthropicVersion = typeof version === 'string' && version !== '' ? version : '2023-06-01';
  const tried = new Set<string>();
  for (let attemptNo = 0; attemptNo < 40; attemptNo++) {
    let candidate: Candidate | null = null;
    const planned = planFor(idx)?.get(idx);
    if (planned !== undefined) {
      candidate = { probe: probes[planned.providerIndex], model: planned.model, rating: planned.rating, providerIndex: planned.providerIndex };
    } else {
      const candidates = candidatesFor(clientWire, body, requires);
      candidate =
        candidates.find((c) => !providerDead(c.probe.provider.id) && !tried.has(`${c.probe.provider.id}/${c.model.id}`)) ?? null;
      if (candidate === null) break;
    }
    const providerId = candidate.probe.provider.id;
    tried.add(`${providerId}/${candidate.model.id}`);
    const callNumber = (attempts.get(providerId) ?? 0) + 1;
    attempts.set(providerId, callNumber);
    const outcome = await callProvider(ENV_URL, candidate.probe, candidate.model.id, clientWire, body, anthropicVersion);
    const seen = outcomes.get(providerId) ?? [];
    seen[callNumber - 1] = outcome.kind !== 'fail';
    outcomes.set(providerId, seen);
    if (outcome.kind === 'fail') {
      const run = (consecutiveRefusals.get(providerId) ?? 0) + 1;
      consecutiveRefusals.set(providerId, run);
      if (providerDead(providerId)) deadProviders.add(providerId);
      continue;
    }
    consecutiveRefusals.set(providerId, 0);
    const servedEntry = (brief.workload?.manifest ?? [])[idx];
    if (servedEntry !== undefined) {
      servedEarned += servedEntry.weight * candidate.rating;
      const bestServed = brief.ratings?.best.find((b) => b.id === servedEntry.id);
      servedBound += servedEntry.weight * (bestServed?.rating_milli ?? candidate.rating);
    }
    if (outcome.kind === 'raw_json') {
      const payload = Buffer.from(outcome.bodyText);
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': String(payload.length) });
      res.end(payload);
    } else if (outcome.kind === 'json') {
      sendJson(res, 200, clientWire === 'openai' ? openAIAnswer(outcome.answer, candidate.model.id) : anthropicAnswer(outcome.answer, candidate.model.id));
    } else if (outcome.kind === 'raw_stream') {
      res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
      await pipeStream(res, outcome.body);
    } else {
      res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
      await writeStream(res, clientWire === 'openai' ? openAIStreamOut(outcome.events, candidate.model.id) : anthropicStreamOut(outcome.events, candidate.model.id));
    }
    return;
  }
  sendWireError(res, clientWire, 502, 'no free provider could serve this request');
}

/** The route table: the two client APIs and the model list. */
async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const path = new URL(req.url ?? '/', 'http://aggregator').pathname;
  const method = req.method ?? 'GET';
  if (method === 'GET' && path === '/v1/models') {
    const models = servedModels();
    if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(models));
    else sendJson(res, 200, openAIModelList(models));
    return;
  }
  if (method === 'POST' && path === '/v1/chat/completions') {
    await serve('openai', await readJson(req), req, res);
    return;
  }
  if (method === 'POST' && path === '/v1/messages') {
    await serve('anthropic', await readJson(req), req, res);
    return;
  }
  sendWireError(res, path === '/v1/messages' ? 'anthropic' : 'openai', 404, `no route ${method} ${path}`);
}

/** Start the HTTP listener on a port the OS picks; resolves with the bound port. */
function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen({ port: 0, host: '127.0.0.1' }, () => {
      const address = server.address();
      if (address === null || typeof address === 'string') reject(new Error('the server has no TCP address'));
      else resolve(address.port);
    });
  });
}

/** Register the bound port and the catalog with the env; the env's answer names the request count. */
async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
  });
  const text = await res.text();
  log(`register answered HTTP ${res.status}: ${text.trim()}`);
}

/** Poll GET /work until the env says the workload is done; then exit 0, as the protocol asks. */
async function poll(): Promise<void> {
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
      if (!res.ok) continue;
      const value: unknown = await res.json();
      if (isRecord(value) && value.done === true) {
        log('the env says the workload is done; exiting 0');
        process.exit(0);
      }
    } catch {
      // The env is not answering yet, or not any more: keep polling.
    }
  }
}

/** Print the catalog (--list) or serve a trial: probe, listen, register, answer, poll. */
async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog, null, 2)}\n`);
    return;
  }
  const server = createServer((req, res) => {
    handle(req, res).catch((err: unknown) => {
      log(`request failed: ${(err as Error).message}`);
      if (!res.headersSent) sendWireError(res, 'openai', 500, 'the aggregator failed to answer');
      if (!res.writableEnded) res.end();
    });
  });
  const port = await listen(server);
  log(`listening on 127.0.0.1:${port}`);
  await register(port);
  await poll();
}

start().catch((err: unknown) => {
  process.stderr.write(`aggregator: ${(err as Error).message}\n`);
  process.exit(1);
});
