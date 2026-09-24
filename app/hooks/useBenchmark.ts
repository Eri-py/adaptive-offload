import { useCallback, useRef, useState } from 'react';
import { Asset } from 'expo-asset';
import { File } from 'expo-file-system';
import { decode as decodeJpeg } from 'jpeg-js';
import { loadTensorflowModel, type TfliteModel } from 'react-native-fast-tflite';

// YOLOv8n's exported TFLite input tensor is [1, 640, 640, 3] float32 NHWC
// (verified against the real exported model in Task 1).
const MODEL_INPUT_SIZE = 640;
const MODEL_INPUT_CHANNELS = 3;

// Metro's bundler needs static, literal `require(...)` calls -- it can't
// resolve a dynamically constructed path string -- so the 15 bundled COCO
// images (Task 1) are listed explicitly here, matching the exact filenames
// present in `app/assets/images/`.
const BUNDLED_IMAGE_MODULES: number[] = [
  require('../assets/images/000000000139.jpg'),
  require('../assets/images/000000000285.jpg'),
  require('../assets/images/000000000632.jpg'),
  require('../assets/images/000000000724.jpg'),
  require('../assets/images/000000000776.jpg'),
  require('../assets/images/000000000785.jpg'),
  require('../assets/images/000000000802.jpg'),
  require('../assets/images/000000000872.jpg'),
  require('../assets/images/000000000885.jpg'),
  require('../assets/images/000000001000.jpg'),
  require('../assets/images/000000001268.jpg'),
  require('../assets/images/000000001296.jpg'),
  require('../assets/images/000000001353.jpg'),
  require('../assets/images/000000001425.jpg'),
  require('../assets/images/000000001490.jpg'),
];

export type BenchmarkStatus = 'idle' | 'running' | 'done' | 'error';

export interface BenchmarkStats {
  mean: number;
  median: number;
  min: number;
  max: number;
  stdev: number;
}

export interface UseBenchmarkResult {
  status: BenchmarkStatus;
  results: BenchmarkStats | null;
  error: string | null;
  runBenchmark: () => Promise<void>;
}

/**
 * Loads the bundled YOLOv8n TFLite model once, then runs on-device inference
 * over all 15 bundled COCO images, timing each `model.run()` call with
 * `performance.now()` (matching the wall-clock-around-inference approach
 * `training/datagen/simulate/yolo_inference.py` uses on the Python side, so
 * "latency" means the same thing on both sides of this project).
 */
export function useBenchmark(): UseBenchmarkResult {
  const [status, setStatus] = useState<BenchmarkStatus>('idle');
  const [results, setResults] = useState<BenchmarkStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const modelRef = useRef<TfliteModel | null>(null);

  const runBenchmark = useCallback(async () => {
    setStatus('running');
    setError(null);
    setResults(null);

    try {
      let model = modelRef.current;
      if (model == null) {
        model = await loadTensorflowModel(require('../assets/models/yolov8n.tflite'), []);
        // Discard a warmup inference: the first real call otherwise pays for
        // one-off weight packing/allocation (and Core ML compilation, if that
        // delegate is on) on top of actual inference, inflating the stats
        // (see review finding B1, matching the Python-side fix).
        const warmupInput = new Float32Array(
          MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * MODEL_INPUT_CHANNELS
        );
        await model.run([warmupInput.buffer as ArrayBuffer]);
        modelRef.current = model;
      }

      const latenciesMs: number[] = [];
      for (const moduleId of BUNDLED_IMAGE_MODULES) {
        const input = await preprocessImage(moduleId);
        const startedAt = performance.now();
        // `Float32Array.prototype.buffer` is typed as `ArrayBufferLike`
        // (it could theoretically back onto a `SharedArrayBuffer`), but this
        // array is always freshly allocated by `resizeAndNormalize` above,
        // so it's always backed by a plain `ArrayBuffer`.
        await model.run([input.buffer as ArrayBuffer]);
        const finishedAt = performance.now();
        latenciesMs.push(finishedAt - startedAt);
      }

      setResults(computeStats(latenciesMs));
      setStatus('done');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus('error');
    }
  }, []);

  return { status, results, error, runBenchmark };
}

/**
 * Resolves a bundled image asset to a correctly-shaped, correctly-normalized
 * Float32Array ready to feed to the model: decodes the JPEG, resizes
 * (nearest-neighbor) to 640x640, drops the alpha channel, and normalizes
 * RGB values to [0, 1]. This benchmark only measures latency (never compares
 * model output), so exact resize/normalization parity with the Python-side
 * preprocessing isn't required -- only a correct input shape and dtype.
 */
async function preprocessImage(moduleId: number): Promise<Float32Array> {
  const asset = Asset.fromModule(moduleId);
  await asset.downloadAsync();
  if (asset.localUri == null) {
    throw new Error(`Could not resolve a local file for bundled image module ${moduleId}`);
  }

  const file = new File(asset.localUri);
  const jpegBytes = await file.arrayBuffer();
  const decoded = decodeJpeg(new Uint8Array(jpegBytes), { useTArray: true });

  return resizeAndNormalize(decoded.data, decoded.width, decoded.height);
}

function resizeAndNormalize(
  sourcePixels: Uint8Array,
  sourceWidth: number,
  sourceHeight: number
): Float32Array {
  const sourceChannels = sourcePixels.length / (sourceWidth * sourceHeight);
  const output = new Float32Array(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * MODEL_INPUT_CHANNELS);

  for (let y = 0; y < MODEL_INPUT_SIZE; y++) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor((y * sourceHeight) / MODEL_INPUT_SIZE));
    for (let x = 0; x < MODEL_INPUT_SIZE; x++) {
      const sourceX = Math.min(sourceWidth - 1, Math.floor((x * sourceWidth) / MODEL_INPUT_SIZE));
      const sourceIndex = (sourceY * sourceWidth + sourceX) * sourceChannels;
      const destIndex = (y * MODEL_INPUT_SIZE + x) * MODEL_INPUT_CHANNELS;

      output[destIndex] = sourcePixels[sourceIndex] / 255;
      output[destIndex + 1] = sourcePixels[sourceIndex + 1] / 255;
      output[destIndex + 2] = sourcePixels[sourceIndex + 2] / 255;
    }
  }

  return output;
}

function computeStats(valuesMs: number[]): BenchmarkStats {
  const sorted = [...valuesMs].sort((a, b) => a - b);
  const count = sorted.length;
  const mean = sorted.reduce((sum, value) => sum + value, 0) / count;
  const mid = Math.floor(count / 2);
  const median = count % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  const variance = sorted.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count;

  return {
    mean,
    median,
    min: sorted[0],
    max: sorted[count - 1],
    stdev: Math.sqrt(variance),
  };
}
