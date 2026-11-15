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
//   quality (60%): every candidate — a free model, on a provider that answered its catalog probe,
//     meeting the request's requirements — is ranked by the quality table's rating, best first.
//     Ties prefer the provider that speaks the client's own wire (its answer then passes through
//     untouched), then the brief's provider order, then the model id, so the order is a pure
//     function of the brief and the listings.
//   failover (10%) and served (15%): a candidate whose call is refused (402, 429, a 5xx, a dead
//     connection) is counted and the request moves to the next candidate, until one answers or
//     none remain. Calls are strictly one at a time. A provider that refuses three calls in a
//     row is demoted behind every other candidate for the rest of the trial — a spent budget and
//     an outage both refuse forever, while the providers' scheduled one-off refusals never reach
//     three in a row — and it is still used when nothing else can serve the request.
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

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Each provider's run of consecutive refused calls this trial; a success resets its run to zero. */
const consecutiveRefusals = new Map<string, number>();

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
  const requires = requirementsFor(body);
  const candidates = candidatesFor(clientWire, body, requires);
  const version = req.headers['anthropic-version'];
  const anthropicVersion = typeof version === 'string' && version !== '' ? version : '2023-06-01';
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    const outcome = await callProvider(ENV_URL, candidate.probe, candidate.model.id, clientWire, body, anthropicVersion);
    if (outcome.kind === 'fail') {
      consecutiveRefusals.set(providerId, (consecutiveRefusals.get(providerId) ?? 0) + 1);
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
