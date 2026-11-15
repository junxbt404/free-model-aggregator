// The wires: the two client APIs this server speaks, the three provider APIs it dials, and every
// translation between them.
//
// The trial's client calls the server as an OpenAI service (POST /v1/chat/completions) or as an
// Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. Providers speak
// one of three wires: OpenAI, Anthropic or Gemini. When the provider's wire is the client's wire
// the request and the answer pass through untouched (only the model id is ours to choose), which
// keeps every field exactly as its owner wrote it. When the wires differ, the request is rebuilt
// in the provider's wire and the answer is rebuilt in the client's, through one normalized form:
//
//   NormalizedAnswer  the text, the tool calls, the stop family and the usage of an answer
//   NormEvent         the same, piece by piece, for a streamed answer
//
// The translations are written against what each wire's parser actually reads out of a request —
// the system prompt's text, the last user message's text, the tool names in order, max_tokens, the
// stop list, the image bytes and the JSON-mode flag — so a rebuilt request still reads identically
// to the one the client sent. Two corners matter:
//
//   - a conversation that ends in tool results reads, on the Anthropic and Gemini parsers, as an
//     empty last user text, while the OpenAI parser keeps the last real user text (and the other
//     way around); the builders below repair that by carrying the last user text along explicitly;
//   - JSON mode exists on the OpenAI wire as response_format and on the Gemini wire as
//     generationConfig.responseMimeType, but the Anthropic wire cannot carry it at all, so the
//     router never sends a JSON-mode request to an Anthropic-wire provider (see main.ts).
//
// Everything here is a pure function of its arguments: fixed timestamps, no clock, no randomness,
// so a replay rebuilds the same bytes.

import { isRecord, type Json } from './providers.ts';

/** One model the server serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** A fixed epoch second for a built completion's `created`. */
export const FIXED_CREATED = 1767225600;

/** GET /v1/models answered to an Authorization: Bearer request: the OpenAI list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models answered to an x-api-key request: the Anthropic list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error body in Anthropic's envelope. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error body in OpenAI's envelope. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/**
 * Why an answer stopped, as a family shared by all three wires:
 * 0 the model finished, 1 it ran out of output room, 2 it called tools, 3 it was filtered.
 */
export type StopFamily = 0 | 1 | 2 | 3;

export function familyOfOpenAI(finish: unknown): StopFamily {
  if (finish === 'length') return 1;
  if (finish === 'tool_calls' || finish === 'function_call') return 2;
  if (finish === 'content_filter') return 3;
  return 0;
}

export function familyOfAnthropic(stop: unknown): StopFamily {
  if (stop === 'max_tokens') return 1;
  if (stop === 'tool_use') return 2;
  if (stop === 'refusal') return 3;
  return 0;
}

export function familyOfGemini(reason: unknown): StopFamily {
  if (reason === 'MAX_TOKENS') return 1;
  if (reason === 'SAFETY' || reason === 'PROHIBITED_CONTENT' || reason === 'BLOCKLIST') return 3;
  return 0;
}

/** A stop family spelled as an OpenAI finish_reason. */
export function openAIFinish(family: StopFamily): string {
  return family === 1 ? 'length' : family === 2 ? 'tool_calls' : family === 3 ? 'content_filter' : 'stop';
}

/** A stop family spelled as an Anthropic stop_reason. */
export function anthropicStop(family: StopFamily): string {
  return family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : family === 3 ? 'refusal' : 'end_turn';
}

/** One tool call of a normalized answer; `args` is the call's arguments as one JSON string. */
export interface NormToolCall {
  id: string;
  name: string;
  args: string;
}

/** An answer with its wire stripped off: text, tool calls, stop family and token usage. */
export interface NormalizedAnswer {
  text: string;
  toolCalls: NormToolCall[];
  family: StopFamily;
  usage: { input: number; output: number };
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object: the JSON string parsed, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// Request builders: a client's request rebuilt in a provider's wire.
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOfAnthropicBlock(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 *
 * One repair: a user message holding only tool_result blocks becomes tool messages and no user
 * message, which would leave the OpenAI parser reading an earlier user text where the Anthropic
 * parser read none. An empty user message is appended in that case, so both parsers agree the
 * last user text is empty.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  let lastUserHadParts = true;
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      if (!assistant) lastUserHadParts = true;
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = partOfAnthropicBlock(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
    lastUserHadParts = parts.length > 0;
  }
  if (!lastUserHadParts) messages.push({ role: 'user', content: '' });

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
    else if (choice.type === 'none') out.tool_choice = 'none';
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The OpenAI parser's reading of a request's system text: each system message's text, joined by newlines. */
function openAISystemText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'system') continue;
    const text = textOf(message.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** The OpenAI parser's reading of a request's last user text (tool and assistant messages do not count). */
function openAILastUserText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let text = '';
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'user') continue;
    text = textOf(message.content);
  }
  return text;
}

/** One OpenAI content part as an Anthropic block: text or image; null for the rest. */
function anthropicBlockOfPart(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
    if (data !== null && data[2] === ';base64') {
      return { type: 'image', source: { type: 'base64', media_type: data[1] ?? 'image/png', data: data[3] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: the system
 * messages join into the system prompt, image_url parts become image blocks, tool_calls become
 * tool_use blocks, tool messages become tool_result blocks in a user message, and the output cap,
 * temperature, top_p, stop list and streaming pass through.
 *
 * One repair: when tool messages trail the conversation, the last Anthropic user message holds
 * only tool_result blocks and its parser reads an empty user text, where the OpenAI parser read
 * the last real user text. The last user text is then appended to that message as a text block,
 * so both parsers agree.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'tool' || raw.role === 'function') {
      const block: Json = {
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
      };
      const last = messages[messages.length - 1];
      if (isRecord(last) && last.role === 'user' && Array.isArray(last.content) && last.content.every((b) => isRecord(b) && b.type === 'tool_result')) {
        last.content = [...last.content, block];
      } else {
        messages.push({ role: 'user', content: [block] });
      }
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') content.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const block = anthropicBlockOfPart(part);
        if (block !== null) content.push(block);
      }
    }
    if (assistant && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
    }
    if (content.length > 0 || !assistant) messages.push({ role: assistant ? 'assistant' : 'user', content });
    else if (assistant) messages.push({ role: 'assistant', content });
  }
  // The trailing tool_result repair described above.
  const lastUserText = openAILastUserText(body);
  const last = messages[messages.length - 1];
  if (
    lastUserText !== '' &&
    isRecord(last) &&
    last.role === 'user' &&
    Array.isArray(last.content) &&
    last.content.length > 0 &&
    last.content.every((b) => isRecord(b) && b.type === 'tool_result')
  ) {
    last.content = [...last.content, { type: 'text', text: lastUserText }];
  }

  const out: Json = { model, messages };
  const system = openAISystemText(body);
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : [];
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as a Gemini part: text, an inline image or a linked image; null for the rest. */
function geminiPartOfPart(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
    if (data !== null && data[2] === ';base64') return { inlineData: { mimeType: data[1] ?? 'image/png', data: data[3] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request for `model` (the model
 * itself travels in the route, not the body): system messages join into systemInstruction, user
 * and assistant turns become user and model contents, tool calls become functionCall parts, tool
 * messages become functionResponse parts, tools become functionDeclarations, and the output cap,
 * temperature, top_p and stop list move into generationConfig. A response_format of json_object or
 * json_schema becomes generationConfig.responseMimeType, the one way the Gemini wire carries JSON
 * mode. The trailing tool-result repair of openAIToAnthropic applies here the same way.
 */
export function openAIToGemini(body: Json, _model: string): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const toolNames = new Map<string, string>();
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'tool' || raw.role === 'function') {
      const part: Json = {
        functionResponse: {
          name: (typeof raw.tool_call_id === 'string' && toolNames.get(raw.tool_call_id)) || 'tool',
          response: { result: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) },
        },
      };
      const last = contents[contents.length - 1];
      if (isRecord(last) && last.role === 'user' && Array.isArray(last.parts) && last.parts.every((p) => isRecord(p) && isRecord(p.functionResponse))) {
        last.parts = [...last.parts, part];
      } else {
        contents.push({ role: 'user', parts: [part] });
      }
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const converted = geminiPartOfPart(part);
        if (converted !== null) parts.push(converted);
      }
    }
    if (raw.role === 'assistant' && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        if (typeof call.id === 'string' && typeof call.function.name === 'string') toolNames.set(call.id, call.function.name);
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
    }
    if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const lastUserText = openAILastUserText(body);
  const last = contents[contents.length - 1];
  if (
    lastUserText !== '' &&
    isRecord(last) &&
    last.role === 'user' &&
    Array.isArray(last.parts) &&
    last.parts.length > 0 &&
    last.parts.every((p) => isRecord(p) && isRecord(p.functionResponse))
  ) {
    last.parts = [...last.parts, { text: lastUserText }];
  }

  const out: Json = { contents };
  const system = openAISystemText(body);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : [];
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  return out;
}

/** An Anthropic Messages request as a Gemini request: through the OpenAI shape, which loses nothing either parser reads. */
export function anthropicToGemini(body: Json, model: string): Json {
  return openAIToGemini(anthropicToOpenAI(body, model), model);
}

// ---------------------------------------------------------------------------
// Answer readers: a provider's answer, in its own wire, as a NormalizedAnswer.
// ---------------------------------------------------------------------------

/** A Chat Completions answer as a NormalizedAnswer. */
export function openAIAnswerOf(body: Json): NormalizedAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: NormToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      args: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    family: familyOfOpenAI(choice.finish_reason),
    usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
  };
}

/** A Messages answer as a NormalizedAnswer. */
export function anthropicAnswerOf(body: Json): NormalizedAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: NormToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        args: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    family: familyOfAnthropic(body.stop_reason),
    usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
  };
}

/** A generateContent answer as a NormalizedAnswer. */
export function geminiAnswerOf(body: Json): NormalizedAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: NormToolCall[] = [];
  let toolIndex = 0;
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolIndex += 1;
      toolCalls.push({
        id: `toolu_gemini_${toolIndex}`,
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        args: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    toolCalls,
    family: familyOfGemini(candidate.finishReason),
    usage: { input: numberOrZero(usage.promptTokenCount), output: numberOrZero(usage.candidatesTokenCount) },
  };
}

// ---------------------------------------------------------------------------
// Answer writers: a NormalizedAnswer rebuilt in the client's wire (JSON form).
// ---------------------------------------------------------------------------

/** A NormalizedAnswer as a Chat Completions object naming the model that served it. */
export function openAIAnswer(answer: NormalizedAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text !== '' || answer.toolCalls.length === 0 ? answer.text : null };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({
      id: call.id !== '' ? call.id : 'call_aggregator',
      type: 'function',
      function: { name: call.name, arguments: call.args },
    }));
  }
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinish(answer.family), logprobs: null }],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  };
}

/** A NormalizedAnswer as a Messages object naming the model that served it. */
export function anthropicAnswer(answer: NormalizedAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_aggregator', name: call.name, input: argumentsOf(call.args) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(answer.family),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/** One server-sent event frame: an `event:` line when named, the JSON data line, the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// Streaming: a provider's event stream read as NormEvents, and NormEvents
// written out as a client's event stream.
// ---------------------------------------------------------------------------

/** One piece of a streamed answer, with its wire stripped off. */
export type NormEvent =
  | { type: 'text'; text: string }
  | { type: 'tool_start'; key: string; id: string; name: string }
  | { type: 'tool_args'; key: string; text: string }
  | { type: 'finish'; family: StopFamily; usage: { input: number; output: number } };

/** The frames of one SSE stream: each frame's event name (or null) and its joined data payload. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<{ event: string | null; data: string }> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const framesOf = (text: string): { event: string | null; data: string }[] => {
    const out: { event: string | null; data: string }[] = [];
    for (const frame of text.split('\n\n')) {
      if (frame.trim() === '') continue;
      let event: string | null = null;
      const data: string[] = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
        else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
      }
      if (data.length > 0) out.push({ event, data: data.join('\n') });
    }
    return out;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const piece = buffer.slice(0, at + 2);
        buffer = buffer.slice(at + 2);
        for (const frame of framesOf(piece)) yield frame;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const frame of framesOf(buffer)) yield frame;
  } finally {
    reader.releaseLock();
  }
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/**
 * A Chat Completions event stream as NormEvents. The finish event is held until the stream's own
 * usage chunk has arrived (the wire sends it after the finish chunk), so the finish NormEvent
 * always carries the stream's real usage.
 */
export async function* openAIStreamEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<NormEvent> {
  let family: StopFamily | null = null;
  let usage: { input: number; output: number } | null = null;
  const startedTools = new Set<string>();
  for await (const frame of sseFrames(stream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    const chunk = jsonOrNull(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') yield { type: 'text', text: delta.content };
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const key = `oai-${typeof call.index === 'number' ? call.index : startedTools.size}`;
      const fn = isRecord(call.function) ? call.function : {};
      if (!startedTools.has(key)) {
        startedTools.add(key);
        yield {
          type: 'tool_start',
          key,
          id: typeof call.id === 'string' && call.id !== '' ? call.id : `call_${key}`,
          name: typeof fn.name === 'string' ? fn.name : '',
        };
      }
      if (typeof fn.arguments === 'string' && fn.arguments !== '') yield { type: 'tool_args', key, text: fn.arguments };
    }
    if (typeof choice.finish_reason === 'string') family = familyOfOpenAI(choice.finish_reason);
  }
  yield { type: 'finish', family: family ?? 0, usage: usage ?? { input: 0, output: 0 } };
}

/** A Messages event stream as NormEvents. */
export async function* anthropicStreamEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<NormEvent> {
  let inputTokens = 0;
  let outputTokens = 0;
  let family: StopFamily = 0;
  let finished = false;
  for await (const frame of sseFrames(stream)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      yield {
        type: 'tool_start',
        key: `ant-${typeof data.index === 'number' ? data.index : 0}`,
        id: typeof data.content_block.id === 'string' ? data.content_block.id : '',
        name: typeof data.content_block.name === 'string' ? data.content_block.name : '',
      };
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        yield { type: 'text', text: data.delta.text };
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && data.delta.partial_json !== '') {
        yield { type: 'tool_args', key: `ant-${typeof data.index === 'number' ? data.index : 0}`, text: data.delta.partial_json };
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta)) family = familyOfAnthropic(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      finished = true;
      yield { type: 'finish', family, usage: { input: inputTokens, output: outputTokens } };
    }
  }
  if (!finished) yield { type: 'finish', family, usage: { input: inputTokens, output: outputTokens } };
}

/** A Gemini event stream as NormEvents: its frames carry whole parts, so a tool call arrives at once. */
export async function* geminiStreamEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<NormEvent> {
  let family: StopFamily = 0;
  let usage = { input: 0, output: 0 };
  let toolCount = 0;
  for await (const frame of sseFrames(stream)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (isRecord(data.usageMetadata)) {
      usage = { input: numberOrZero(data.usageMetadata.promptTokenCount), output: numberOrZero(data.usageMetadata.candidatesTokenCount) };
    }
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    if (typeof candidate.finishReason === 'string') family = familyOfGemini(candidate.finishReason);
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string' && part.text !== '') yield { type: 'text', text: part.text };
      else if (isRecord(part.functionCall)) {
        toolCount += 1;
        const key = `gem-${toolCount}`;
        yield {
          type: 'tool_start',
          key,
          id: `toolu_gemini_${toolCount}`,
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        };
        yield { type: 'tool_args', key, text: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) };
      }
    }
  }
  yield { type: 'finish', family, usage };
}

/** NormEvents written as a Chat Completions event stream naming the model that served it. */
export async function* openAIStreamOut(events: AsyncIterable<NormEvent>, model: string): AsyncGenerator<string> {
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
  let roleSent = false;
  let finished = false;
  const toolIndexes = new Map<string, number>();
  for await (const event of events) {
    if (!roleSent) {
      roleSent = true;
      yield chunk({ role: 'assistant', content: '' }, null);
    }
    if (event.type === 'text') {
      yield chunk({ content: event.text }, null);
    } else if (event.type === 'tool_start') {
      const index = toolIndexes.size;
      toolIndexes.set(event.key, index);
      yield chunk(
        { tool_calls: [{ index, id: event.id !== '' ? event.id : `call_${index}`, type: 'function', function: { name: event.name, arguments: '' } }] },
        null,
      );
    } else if (event.type === 'tool_args') {
      const index = toolIndexes.get(event.key) ?? 0;
      yield chunk({ tool_calls: [{ index, function: { arguments: event.text } }] }, null);
    } else if (event.type === 'finish') {
      finished = true;
      yield chunk({}, openAIFinish(event.family));
      yield sseEvent(null, {
        id: 'chatcmpl-aggregator',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: {
          prompt_tokens: event.usage.input,
          completion_tokens: event.usage.output,
          total_tokens: event.usage.input + event.usage.output,
        },
      });
      yield 'data: [DONE]\n\n';
    }
  }
  if (!finished) {
    if (!roleSent) yield chunk({ role: 'assistant', content: '' }, null);
    yield chunk({}, 'stop');
    yield 'data: [DONE]\n\n';
  }
}

/** NormEvents written as a Messages event stream naming the model that served it. */
export async function* anthropicStreamOut(events: AsyncIterable<NormEvent>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<string, number>();
  const start = (): string =>
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });
  const closeText = (): string | null => {
    if (textIndex === -1) return null;
    const index = textIndex;
    textIndex = -1;
    return sseEvent('content_block_stop', { type: 'content_block_stop', index });
  };
  for await (const event of events) {
    if (!started) {
      started = true;
      yield start();
    }
    if (event.type === 'text') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: event.text } });
    } else if (event.type === 'tool_start') {
      const closed = closeText();
      if (closed !== null) yield closed;
      const index = nextIndex;
      nextIndex += 1;
      toolIndexes.set(event.key, index);
      yield sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: event.id !== '' ? event.id : `toolu_${index}`, name: event.name, input: {} },
      });
    } else if (event.type === 'tool_args') {
      const index = toolIndexes.get(event.key);
      if (index !== undefined) {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: event.text } });
      }
    } else if (event.type === 'finish') {
      finished = true;
      const closed = closeText();
      if (closed !== null) yield closed;
      for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
        yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
      }
      toolIndexes.clear();
      yield sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: anthropicStop(event.family), stop_sequence: null },
        usage: { output_tokens: event.usage.output },
      });
      yield sseEvent('message_stop', { type: 'message_stop' });
    }
  }
  if (!finished) {
    if (!started) yield start();
    const closed = closeText();
    if (closed !== null) yield closed;
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
    }
    yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 0 } });
    yield sseEvent('message_stop', { type: 'message_stop' });
  }
}
