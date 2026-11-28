import { useCallback, useEffect, useRef, useState } from 'react';
import { preloadLocalModel, runLocalInference } from '../services/localInference';
import { classifyImage } from '../services/offloadClient';
import type { OffloadFailure } from '../services/offloadClient';
import type { ChosenPhoto } from '../services/pickPhoto';
import { prepareImage } from '../services/prepareImage';
import type { PreparedImage } from '../services/prepareImage';

export type PathName = 'local' | 'offload';

// Both paths normalise to this so one component can render either.
export interface PathResult {
  species: string;
  classIndex: number;
  maxProbability: number;
  top2Margin: number;
  top3: { index: number; name: string; probability: number }[];
  // What the user waits for: local decode+tensor+model, or the offload round trip.
  latencyMs: number;
  // Model-only time: on-device model.run, or the server's own compute.
  // Remainder is latencyMs - modelMs: decode/tensor locally, network on offload.
  modelMs: number;
}

export type PathState =
  | { status: 'idle' }
  | { status: 'running' }
  | { status: 'done'; result: PathResult }
  | { status: 'failed'; reason: string };

export type LatencyWinner = PathName | 'unavailable';
// 'unknown': no true species. 'unavailable': a path failed. 'both'/'neither' cover ties and double misses.
export type AccuracyWinner = PathName | 'both' | 'neither' | 'unknown' | 'unavailable';

export interface ScanSummary {
  latencyWinner: LatencyWinner;
  accuracyWinner: AccuracyWinner;
}

export type ScanPhase = 'idle' | 'preparing' | 'running' | 'settled' | 'prepare-failed';

export interface ScanState {
  phase: ScanPhase;
  photo: ChosenPhoto | null;
  prepared: PreparedImage | null;
  byteCount: number | null;
  prepareError: string | null;
  local: PathState;
  offload: PathState;
  summary: ScanSummary | null;
}

export interface UseScan extends ScanState {
  startScan: (photo: ChosenPhoto) => Promise<void>;
  reset: () => void;
}

const INITIAL: ScanState = {
  phase: 'idle',
  photo: null,
  prepared: null,
  byteCount: null,
  prepareError: null,
  local: { status: 'idle' },
  offload: { status: 'idle' },
  summary: null,
};

function describeOffloadFailure(failure: OffloadFailure): string {
  switch (failure.kind) {
    case 'timeout':
      return `Server did not answer in time (${failure.message}).`;
    case 'network':
      return `Server unreachable: ${failure.message}`;
    case 'http':
      return `Server returned ${failure.status}: ${failure.message}`;
    case 'invalid-response':
      return `Server sent an unusable reply: ${failure.message}`;
  }
}

async function runLocalPath(bytes: Uint8Array): Promise<PathState> {
  try {
    const r = await runLocalInference(bytes);
    return {
      status: 'done',
      result: {
        species: r.species,
        classIndex: r.class_index,
        maxProbability: r.max_probability,
        top2Margin: r.top2_margin,
        top3: r.top3,
        latencyMs: r.total_ms,
        modelMs: r.compute_ms,
      },
    };
  } catch (error) {
    return { status: 'failed', reason: `On-device inference failed: ${String(error)}` };
  }
}

async function runOffloadPath(bytes: Uint8Array): Promise<PathState> {
  try {
    const outcome = await classifyImage(bytes);
    if (!outcome.ok) return { status: 'failed', reason: describeOffloadFailure(outcome.failure) };
    const { response, roundTripMs, serverComputeMs } = outcome.value;
    return {
      status: 'done',
      result: {
        species: response.species,
        classIndex: response.class_index,
        maxProbability: response.max_probability,
        top2Margin: response.top2_margin,
        top3: response.top3,
        latencyMs: roundTripMs,
        modelMs: serverComputeMs,
      },
    };
  } catch (error) {
    // classifyImage should never throw; this keeps a bug from rejecting the scan.
    return { status: 'failed', reason: `Offload failed unexpectedly: ${String(error)}` };
  }
}

// Reports only; nothing here feeds back into which path runs.
export function summarise(
  local: PathState,
  offload: PathState,
  trueLabel: number | null,
): ScanSummary {
  if (local.status !== 'done' || offload.status !== 'done') {
    return { latencyWinner: 'unavailable', accuracyWinner: trueLabel === null ? 'unknown' : 'unavailable' };
  }
  const l = local.result;
  const o = offload.result;
  // Ties go to local: equal waits mean offloading bought nothing.
  const latencyWinner: PathName = l.latencyMs <= o.latencyMs ? 'local' : 'offload';
  if (trueLabel === null) return { latencyWinner, accuracyWinner: 'unknown' };
  const localRight = l.classIndex === trueLabel;
  const offloadRight = o.classIndex === trueLabel;
  let accuracyWinner: AccuracyWinner;
  if (localRight && offloadRight) accuracyWinner = 'both';
  else if (localRight) accuracyWinner = 'local';
  else if (offloadRight) accuracyWinner = 'offload';
  else accuracyWinner = 'neither';
  return { latencyWinner, accuracyWinner };
}

export function useScan(): UseScan {
  const [state, setState] = useState<ScanState>(INITIAL);
  // Bumped per scan and on reset so a stale scan cannot overwrite a newer one.
  const scanId = useRef(0);

  useEffect(() => {
    // Warm the model now so the first scan does not pay for load and warmup; never throws.
    void preloadLocalModel();
    return () => {
      scanId.current += 1;
    };
  }, []);

  const reset = useCallback((): void => {
    scanId.current += 1;
    setState(INITIAL);
  }, []);

  const startScan = useCallback(async (photo: ChosenPhoto): Promise<void> => {
    scanId.current += 1;
    const id = scanId.current;
    const isCurrent = (): boolean => scanId.current === id;
    const patch = (next: Partial<ScanState>): void => {
      if (isCurrent()) setState((prev) => ({ ...prev, ...next }));
    };

    setState({ ...INITIAL, phase: 'preparing', photo });

    let prepared: PreparedImage;
    try {
      prepared = await prepareImage(photo.uri);
    } catch (error) {
      patch({ phase: 'prepare-failed', prepareError: `Could not prepare the photo: ${String(error)}` });
      return;
    }

    // One view over one buffer: both paths get these exact bytes, with no second conversion.
    const bytes = new Uint8Array(prepared.jpegBytes);
    patch({
      phase: 'running',
      prepared,
      byteCount: prepared.byteCount,
      local: { status: 'running' },
      offload: { status: 'running' },
    });

    // Each promise publishes itself on arrival so local never waits for the server.
    const localDone = runLocalPath(bytes).then((local) => {
      patch({ local });
      return local;
    });
    const offloadDone = runOffloadPath(bytes).then((offload) => {
      patch({ offload });
      return offload;
    });

    const [local, offload] = await Promise.all([localDone, offloadDone]);
    patch({
      phase: 'settled',
      summary: summarise(local, offload, photo.trueSpecies ? photo.trueSpecies.label : null),
    });
  }, []);

  return { ...state, startScan, reset };
}
