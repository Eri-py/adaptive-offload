import { readServerUrl } from './serverConfig';

// Generous: a slow first request over the relay must not look like an outage.
export const REQUEST_TIMEOUT_MS = 5000;

export type Top3Entry = {
  index: number;
  name: string;
  probability: number;
};

export type ClassifyResponse = {
  species: string;
  class_index: number;
  max_probability: number;
  top2_margin: number;
  top3: Top3Entry[];
  compute_ms: number;
  geometric_transform_applied: boolean;
};

export type HealthResponse = {
  status: string;
  model_loaded: boolean;
};

export type OffloadFailure =
  | { kind: 'timeout'; message: string }
  | { kind: 'network'; message: string }
  | { kind: 'http'; status: number; message: string }
  | { kind: 'invalid-response'; message: string };

export type ClientResult<T> =
  | { ok: true; value: T }
  | { ok: false; failure: OffloadFailure };

export type ClassifyOutcome = {
  response: ClassifyResponse;
  // Phone-measured total round trip; includes network plus server compute.
  roundTripMs: number;
  // The server's own reported compute, kept separate so network cost is roundTripMs - serverComputeMs.
  serverComputeMs: number;
};

function fail<T>(failure: OffloadFailure): ClientResult<T> {
  return { ok: false, failure };
}

async function detailOf(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    const detail = (body as { detail?: unknown } | null)?.detail;
    if (typeof detail === 'string') return detail;
  } catch {
    // Body was not JSON; fall through to the status text.
  }
  return response.statusText || `HTTP ${response.status}`;
}

// Runs one request with a timeout; the timer is always cleared so it cannot fire late.
async function request(
  path: string,
  init: RequestInit,
): Promise<ClientResult<{ response: Response; roundTripMs: number }>> {
  let baseUrl: string;
  try {
    baseUrl = (await readServerUrl()).url.replace(/\/+$/, '');
  } catch (error) {
    return fail({ kind: 'network', message: `Server address unavailable: ${String(error)}` });
  }
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, REQUEST_TIMEOUT_MS);
  const startedAt = Date.now();
  try {
    const response = await fetch(`${baseUrl}${path}`, { ...init, signal: controller.signal });
    if (!response.ok) {
      return fail({ kind: 'http', status: response.status, message: await detailOf(response) });
    }
    // Stop the clock once headers arrive; body parse is local and not part of the trip.
    return { ok: true, value: { response, roundTripMs: Date.now() - startedAt } };
  } catch (error) {
    if (timedOut) {
      return fail({ kind: 'timeout', message: `No response within ${REQUEST_TIMEOUT_MS} ms` });
    }
    return fail({ kind: 'network', message: String(error) });
  } finally {
    clearTimeout(timer);
  }
}

function isHealth(value: unknown): value is HealthResponse {
  const v = value as Partial<HealthResponse> | null;
  return !!v && typeof v.status === 'string' && typeof v.model_loaded === 'boolean';
}

function isClassify(value: unknown): value is ClassifyResponse {
  const v = value as Partial<ClassifyResponse> | null;
  return (
    !!v &&
    typeof v.species === 'string' &&
    typeof v.class_index === 'number' &&
    typeof v.max_probability === 'number' &&
    typeof v.top2_margin === 'number' &&
    Array.isArray(v.top3) &&
    typeof v.compute_ms === 'number' &&
    typeof v.geometric_transform_applied === 'boolean'
  );
}

async function parseJson(response: Response): Promise<ClientResult<unknown>> {
  try {
    return { ok: true, value: await response.json() };
  } catch (error) {
    return fail({ kind: 'invalid-response', message: `Response was not JSON: ${String(error)}` });
  }
}

export async function checkHealth(): Promise<ClientResult<HealthResponse>> {
  const sent = await request('/health', { method: 'GET' });
  if (!sent.ok) return sent;
  const parsed = await parseJson(sent.value.response);
  if (!parsed.ok) return parsed;
  if (!isHealth(parsed.value)) {
    return fail({ kind: 'invalid-response', message: 'Unexpected /health response shape' });
  }
  return { ok: true, value: parsed.value };
}

// Typings accept ArrayBuffer but not views; copy so a sliced view sends only its own bytes.
function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

// Raw JPEG body, not multipart: the research payload figure is the image bytes alone.
export async function classifyImage(jpegBytes: Uint8Array): Promise<ClientResult<ClassifyOutcome>> {
  const sent = await request('/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'image/jpeg' },
    body: toArrayBuffer(jpegBytes),
  });
  if (!sent.ok) return sent;
  const parsed = await parseJson(sent.value.response);
  if (!parsed.ok) return parsed;
  if (!isClassify(parsed.value)) {
    return fail({ kind: 'invalid-response', message: 'Unexpected /classify response shape' });
  }
  return {
    ok: true,
    value: {
      response: parsed.value,
      roundTripMs: sent.value.roundTripMs,
      serverComputeMs: parsed.value.compute_ms,
    },
  };
}
