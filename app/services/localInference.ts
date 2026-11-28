import { decode as decodeJpeg } from 'jpeg-js';
import { loadTensorflowModel, type TfliteModel } from 'react-native-fast-tflite';
import speciesNames from '../assets/models/cub200-species.json';

const INPUT_SIZE = 224;
const INPUT_CHANNELS = 3;
const PIXELS = INPUT_SIZE * INPUT_SIZE;
const NUM_CLASSES = 200;
const SOURCE_STRIDE = 4;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const SPECIES: string[] = speciesNames;

export interface LocalTopEntry {
  index: number;
  name: string;
  probability: number;
}

// Field names mirror the server response so one component can render either path.
export interface LocalInferenceResult {
  species: string;
  class_index: number;
  max_probability: number;
  top2_margin: number;
  top3: LocalTopEntry[];
  compute_ms: number;
  // Decode + tensor build + model.run + softmax; excludes the one-time model load.
  total_ms: number;
}

// Metro needs a literal require() path.
const MODEL_SOURCE: number = require('../assets/models/small.tflite');

// Shared promise so concurrent first scans cannot load the model twice.
let modelPromise: Promise<TfliteModel> | null = null;

function loadModelOnce(): Promise<TfliteModel> {
  if (modelPromise == null) {
    modelPromise = initModel().catch((err: unknown) => {
      // Drop the cached failure so a later scan can retry instead of failing forever.
      modelPromise = null;
      throw err;
    });
  }
  return modelPromise;
}

async function initModel(): Promise<TfliteModel> {
  const model = await loadTensorflowModel(MODEL_SOURCE, ['core-ml']);
  // Discarded warmup so Core ML compile / weight packing stays out of timed runs.
  const warmup = new Float32Array(PIXELS * INPUT_CHANNELS);
  await model.run([warmup.buffer as ArrayBuffer]);
  return model;
}

export type PreloadStatus = { ok: true } | { ok: false; error: string };

// Starts (or joins) the shared load+warmup; resolves with a status and never rejects.
export async function preloadLocalModel(): Promise<PreloadStatus> {
  try {
    await loadModelOnce();
    return { ok: true };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { ok: false, error: `On-device model preload failed: ${message}` };
  }
}

// Runs the on-device model on prepared 224x224 JPEG bytes; compute_ms covers model.run() only.
export async function runLocalInference(
  jpegBytes: ArrayBuffer | Uint8Array
): Promise<LocalInferenceResult> {
  try {
    const model = await loadModelOnce();
    // Started after load: that is a one-time cost, not part of a scan.
    const pathStartedAt = performance.now();
    const bytes = jpegBytes instanceof Uint8Array ? jpegBytes : new Uint8Array(jpegBytes);
    const decoded = decodeJpeg(bytes, { useTArray: true });
    if (decoded.width !== INPUT_SIZE || decoded.height !== INPUT_SIZE) {
      throw new Error(`Prepared image is ${decoded.width}x${decoded.height}, expected ${INPUT_SIZE}`);
    }
    const input = buildInputTensor(decoded.data);

    const startedAt = performance.now();
    // Cast is safe: freshly allocated, never a SharedArrayBuffer.
    const outputs = await model.run([input.buffer as ArrayBuffer]);
    const computeMs = performance.now() - startedAt;

    const logits = new Float32Array(outputs[0]);
    if (logits.length !== NUM_CLASSES) {
      throw new Error(`Expected ${NUM_CLASSES} logits, got ${logits.length}`);
    }
    const result = summarize(logits, computeMs);
    result.total_ms = performance.now() - pathStartedAt;
    return result;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`On-device inference failed: ${message}`);
  }
}

// ImageNet-normalizes RGBA bytes into planar NCHW (all R, then all G, then all B).
function buildInputTensor(rgba: Uint8Array): Float32Array {
  const output = new Float32Array(PIXELS * INPUT_CHANNELS);
  for (let y = 0; y < INPUT_SIZE; y++) {
    for (let x = 0; x < INPUT_SIZE; x++) {
      const pixelIndex = y * INPUT_SIZE + x;
      const sourceIndex = pixelIndex * SOURCE_STRIDE;
      for (let channel = 0; channel < INPUT_CHANNELS; channel++) {
        output[channel * PIXELS + pixelIndex] =
          (rgba[sourceIndex + channel] / 255 - MEAN[channel]) / STD[channel];
      }
    }
  }
  return output;
}

function summarize(logits: Float32Array, computeMs: number): LocalInferenceResult {
  let maxLogit = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > maxLogit) {
      maxLogit = logits[i];
    }
  }
  // Subtracting the max keeps exp() from overflowing.
  const exps = Array.from(logits, (logit) => Math.exp(logit - maxLogit));
  const sum = exps.reduce((total, value) => total + value, 0);

  const top3 = exps
    .map((value, index) => ({ index, name: SPECIES[index] ?? `class ${index}`, probability: value / sum }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 3);

  return {
    species: top3[0].name,
    class_index: top3[0].index,
    max_probability: top3[0].probability,
    top2_margin: top3[0].probability - top3[1].probability,
    top3,
    compute_ms: computeMs,
    // Filled in by the caller once softmax is done.
    total_ms: computeMs,
  };
}
