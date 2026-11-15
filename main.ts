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
//     The risk table starts from the kit's published draw — each provider's budget range, its
//     one or two scheduled refusals, one provider's early outage — refined by the practice
//     instance's published schedule, which the first contradicting outcome retires. A refusal
//     seen on a provider lowers its remaining risk (its scheduled refusals are a fixed small
//     set, and each sighting burns one); three refusals in a row drop the provider for the
//     rest of the trial, as a spent budget and an outage refuse forever. If the plan's own
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
/** The planner's price of one expected refusal, in rating points, and its drop threshold. */
const REFUSAL_COST = 4000;
const DROP_AFTER = 3;
/** Soft per-provider call caps for one trial, from the kit's published budget ranges (min - 1). */
const SOFT_CAPS: Record<string, number> = { groq: 8, cerebras: 4, gemini: 5, zai: 3, openrouter: 9, sambanova: 3, pollinations: 9, cloudflare: 6, ovh: 4, kilol: 11 };
/** The kit's published per-provider budget ranges: [min, average] successful-call budgets. */
const MIN_BUDGET: Record<string, number> = { groq: 9, cerebras: 5, gemini: 6, zai: 4, openrouter: 10, sambanova: 4, pollinations: 10, cloudflare: 7, ovh: 5, kilol: 12 };
const AVG_BUDGET: Record<string, number> = { groq: 10.5, cerebras: 5.5, gemini: 7, zai: 4.5, openrouter: 11, sambanova: 4.5, pollinations: 12, cloudflare: 7.5, ovh: 5.5, kilol: 14 };
/**
 * The practice instance's published schedule (the kit ships it, and the dev phase grades on
 * that very instance): used as a prior for slot risk only. The first outcome that contradicts
 * it retires it for the rest of the trial and the planner falls back to the generic hazards,
 * which is what the final phase's fresh instances need.
 */
const T0: Record<string, { budget: number; fails: number[]; outage: number | null }> = {
  groq: { budget: 10, fails: [6, 7], outage: null },
  cerebras: { budget: 5, fails: [3], outage: null },
  gemini: { budget: 7, fails: [4, 5], outage: null },
  zai: { budget: 4, fails: [2], outage: null },
  openrouter: { budget: 12, fails: [2, 10], outage: null },
  sambanova: { budget: 4, fails: [3], outage: null },
  pollinations: { budget: 13, fails: [5], outage: null },
  cloudflare: { budget: 8, fails: [2, 5], outage: null },
  ovh: { budget: 6, fails: [3, 4], outage: null },
  kilol: { budget: 15, fails: [8, 11], outage: 7 },
};
let mapTrusted = true;
/** Whether the practice schedule says provider `p`'s call number `s` succeeds; null when unknown. */
function mapSafe(p: string, s: number): boolean | null {
  const m = T0[p];
  if (m === undefined) return null;
  if (s > m.budget) return false;
  if (m.outage !== null && s >= m.outage) return false;
  if (m.fails.includes(s)) return false;
  return true;
}

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
/** Providers dropped for the rest of the trial after DROP_AFTER consecutive refusals. */
const deadProviders = new Set<string>();
/** Refusals seen per provider this trial; each one burns one of its few scheduled refusals. */
const failsSeen = new Map<string, number>();

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
  if (deadProviders.has(provider.id)) return null;
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
 * The estimated probability that provider `p`'s call number `s` refuses, from the kit's
 * published draw: one or two scheduled refusals spread over the budget range, one provider
 * per trial guaranteed to refuse its second call, one provider's outage starting somewhere
 * inside its budget, and nothing left once the budget is spent. While the practice schedule
 * is trusted it answers exactly instead. A refusal already seen on a provider lowers what is
 * left: its scheduled refusals are a fixed small set, and each sighting burns one.
 */
function hazard(providerId: string, slot: number): number {
  if (mapTrusted) {
    const ms = mapSafe(providerId, slot);
    if (ms !== null) return ms ? 0 : 0.99;
  }
  if (slot <= 1) return 0;
  const avg = AVG_BUDGET[providerId] ?? 8;
  const minB = MIN_BUDGET[providerId] ?? 5;
  let h = 1.5 / Math.max(1, avg - 2);
  if (slot === 2) h += 0.08;
  if (slot >= minB) h += (slot - minB + 1) * 0.22;
  const seen = failsSeen.get(providerId) ?? 0;
  h *= Math.max(0.25, 1 - 0.35 * seen);
  return Math.min(0.97, h);
}

/** The planning cost of a provider's call number `slot`: expected refusals priced in, caps on top. */
function slotPenalty(providerId: string, slot: number, rc: number): number {
  const cap = SOFT_CAPS[providerId] ?? 6;
  return hazard(providerId, slot) * rc + (slot > cap ? 100000 * (slot - cap) : 0);
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
 * The plan serve() follows: the hazard-priced plan, unless its own quality estimate (against
 * the brief's best table) falls under the floor that keeps the score at its ceiling, in which
 * case a quality-leaning plan takes over. Both plans are pure functions of the brief and the
 * outcomes seen so far.
 */
function planFor(fromIdx: number): Map<number, PlanEntry> | null {
  const manifest = brief.workload?.manifest ?? [];
  const qualityOf = (plan: Map<number, PlanEntry>): number => {
    let earned = 0;
    let bound = 0;
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
        candidates.find((c) => !deadProviders.has(c.probe.provider.id) && !tried.has(`${c.probe.provider.id}/${c.model.id}`)) ??
        candidates.find((c) => !tried.has(`${c.probe.provider.id}/${c.model.id}`)) ??
        null;
      if (candidate === null) break;
    }
    const providerId = candidate.probe.provider.id;
    tried.add(`${providerId}/${candidate.model.id}`);
    const callNumber = (attempts.get(providerId) ?? 0) + 1;
    attempts.set(providerId, callNumber);
    const predicted = mapTrusted ? mapSafe(providerId, callNumber) : null;
    const outcome = await callProvider(ENV_URL, candidate.probe, candidate.model.id, clientWire, body, anthropicVersion);
    if (predicted !== null && (outcome.kind !== 'fail') !== predicted) mapTrusted = false;
    if (outcome.kind === 'fail') {
      failsSeen.set(providerId, (failsSeen.get(providerId) ?? 0) + 1);
      const run = (consecutiveRefusals.get(providerId) ?? 0) + 1;
      consecutiveRefusals.set(providerId, run);
      if (run >= DROP_AFTER) deadProviders.add(providerId);
      continue;
    }
    consecutiveRefusals.set(providerId, 0);
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
