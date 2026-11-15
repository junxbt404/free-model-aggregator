// Calling the providers: one request, in the provider's own wire, with failover left to the caller.
//
// callProvider builds the request for the routed provider's wire (translating the client's body
// when the wires differ), sends it with the provider's own auth scheme, and classifies the answer:
//
//   raw_json / raw_stream   the provider speaks the client's wire: pass its answer through as it
//                           stands, so every field stays exactly as its owner wrote it;
//   json / stream           the wires differ: the answer comes back normalized (a NormalizedAnswer,
//                           or a stream of NormEvents) for main.ts to rebuild in the client's wire;
//   fail                    any non-2xx answer (a 402 or 429 refusal, a 5xx, anything else): the
//                           caller counts it and tries the next candidate. A fetch that throws
//                           (an unreachable provider, a timeout) is a fail with status 0.
//
// Calls are made one at a time by the caller, so a provider that serves one call at a time never
// sees a second call of ours while its first is in flight.

import { authHeaders, isRecord, joinBase, type Json, type Probe } from './providers.ts';
import {
  anthropicAnswerOf,
  anthropicStreamEvents,
  anthropicToGemini,
  anthropicToOpenAI,
  geminiAnswerOf,
  geminiStreamEvents,
  openAIAnswerOf,
  openAIStreamEvents,
  openAIToAnthropic,
  openAIToGemini,
  type NormEvent,
  type NormalizedAnswer,
} from './wire.ts';

export type ClientWire = 'openai' | 'anthropic';

export type CallOutcome =
  | { kind: 'raw_json'; bodyText: string }
  | { kind: 'raw_stream'; body: ReadableStream<Uint8Array> }
  | { kind: 'json'; answer: NormalizedAnswer }
  | { kind: 'stream'; events: AsyncGenerator<NormEvent> }
  | { kind: 'fail'; status: number };

/** How long one provider call may take before it counts as a failure; the client's own cap is 30 s. */
const CALL_TIMEOUT_MS = 25000;

/** The upstream request for one routed provider: its URL, headers and body in its own wire. */
function buildRequest(
  envUrl: string,
  probe: Probe,
  modelId: string,
  clientWire: ClientWire,
  body: Json,
  anthropicVersion: string,
): { url: string; headers: Record<string, string>; payload: Json } {
  const provider = probe.provider;
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  const stream = body.stream === true;
  if (provider.wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    const payload =
      clientWire === 'openai'
        ? { ...body, model: modelId, ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, modelId);
    return { url: `${base}/v1/chat/completions`, headers, payload };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = anthropicVersion;
    if (stream) headers.accept = 'text/event-stream';
    const payload = clientWire === 'anthropic' ? { ...body, model: modelId } : openAIToAnthropic(body, modelId);
    return { url: `${base}/v1/messages`, headers, payload };
  }
  // The Gemini wire: the model travels in the route, and streaming is a different route altogether.
  if (stream) headers.accept = 'text/event-stream';
  const payload = clientWire === 'openai' ? openAIToGemini(body, modelId) : anthropicToGemini(body, modelId);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const query = stream ? '?alt=sse' : '';
  return { url: `${base}/v1beta/models/${encodeURIComponent(modelId)}:${action}${query}`, headers, payload };
}

/**
 * One call to one provider for one client request. `body` is the client's own request body;
 * `anthropicVersion` is the version header the client sent (or the default), forwarded to
 * Anthropic-wire providers as the real service expects it.
 */
export async function callProvider(
  envUrl: string,
  probe: Probe,
  modelId: string,
  clientWire: ClientWire,
  body: Json,
  anthropicVersion: string,
): Promise<CallOutcome> {
  const wire = probe.provider.wire;
  const sameWire = (wire === 'openai' && clientWire === 'openai') || (wire === 'anthropic' && clientWire === 'anthropic');
  const stream = body.stream === true;
  const { url, headers, payload } = buildRequest(envUrl, probe, modelId, clientWire, body, anthropicVersion);
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return { kind: 'fail', status: 0 };
  }
  if (!res.ok) {
    // Drain the refusal so the connection is free, then let the caller fail over.
    try {
      await res.text();
    } catch {
      // Nothing to drain: the refusal stands either way.
    }
    return { kind: 'fail', status: res.status };
  }
  if (stream) {
    if (res.body === null) return { kind: 'fail', status: 0 };
    if (sameWire) return { kind: 'raw_stream', body: res.body };
    if (wire === 'openai') return { kind: 'stream', events: openAIStreamEvents(res.body) };
    if (wire === 'anthropic') return { kind: 'stream', events: anthropicStreamEvents(res.body) };
    return { kind: 'stream', events: geminiStreamEvents(res.body) };
  }
  const text = await res.text();
  if (sameWire) return { kind: 'raw_json', bodyText: text };
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { kind: 'fail', status: 502 };
  }
  if (!isRecord(parsed)) return { kind: 'fail', status: 502 };
  if (wire === 'openai') return { kind: 'json', answer: openAIAnswerOf(parsed) };
  if (wire === 'anthropic') return { kind: 'json', answer: anthropicAnswerOf(parsed) };
  return { kind: 'json', answer: geminiAnswerOf(parsed) };
}
