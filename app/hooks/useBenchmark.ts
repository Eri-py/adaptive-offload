import { useCallback, useRef, useState } from 'react';
import { Asset } from 'expo-asset';
import { File } from 'expo-file-system';
import { decode as decodeJpeg } from 'jpeg-js';
import {
  loadTensorflowModel,
  type TfliteModel,
  type TensorflowModelDelegate,
} from 'react-native-fast-tflite';

// YOLOv8n's exported TFLite input tensor is [1, 640, 640, 3] float32 NHWC
// (verified against the real exported model in Task 1).
const MODEL_INPUT_SIZE = 640;
const MODEL_INPUT_CHANNELS = 3;

// Metro needs literal require() paths, so images are listed explicitly.
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

// Single source of truth for the screen's "Images" label — keeps it in sync with the array above.
export const BUNDLED_IMAGE_COUNT = BUNDLED_IMAGE_MODULES.length;

export type BenchmarkStatus = 'idle' | 'running' | 'done' | 'error';

export interface BenchmarkStats {
  mean: number;
  median: number;
  min: number;
  max: number;
  stdev: number;
}

export type DelegateId = 'cpu' | 'core-ml';

export interface DelegateResult {
  id: DelegateId;
  label: string;
  stats: BenchmarkStats | null;
  error: string | null;
}

export interface UseBenchmarkResult {
  status: BenchmarkStatus;
  results: DelegateResult[] | null;
  error: string | null;
  runBenchmark: () => Promise<void>;
}

// CPU is the baseline vs. desktop numbers; Core ML is the real on-device path.
const DELEGATE_SPECS: { id: DelegateId; label: string; delegates: TensorflowModelDelegate[] }[] = [
  { id: 'cpu', label: 'CPU', delegates: [] },
  { id: 'core-ml', label: 'Core ML', delegates: ['core-ml'] },
];

// Times only the raw model.run() forward pass on a preprocessed tensor —
// narrower than yolo_inference.py's model(image_path), which also decodes, letterboxes, and runs NMS.
export function useBenchmark(): UseBenchmarkResult {
  const [status, setStatus] = useState<BenchmarkStatus>('idle');
  const [results, setResults] = useState<DelegateResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const modelsRef = useRef<Partial<Record<DelegateId, TfliteModel>>>({});

  const runBenchmark = useCallback(async () => {
    setStatus('running');
    setError(null);
    setResults(null);

    try {
      const delegateResults: DelegateResult[] = [];

      for (const spec of DELEGATE_SPECS) {
        try {
          let model = modelsRef.current[spec.id];
          if (model == null) {
            model = await loadTensorflowModel(
              require('../assets/models/yolov8n.tflite'),
              spec.delegates
            );
            // Discard the warmup run — it pays for weight packing/Core ML compile that would inflate stats.
            const warmupInput = new Float32Array(
              MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * MODEL_INPUT_CHANNELS
            );
            await model.run([warmupInput.buffer as ArrayBuffer]);
            modelsRef.current[spec.id] = model;
          }

          const latenciesMs: number[] = [];
          for (const moduleId of BUNDLED_IMAGE_MODULES) {
            const input = await preprocessImage(moduleId);
            const startedAt = performance.now();
            // Cast is safe: this array is always freshly allocated, never a SharedArrayBuffer.
            await model.run([input.buffer as ArrayBuffer]);
            const finishedAt = performance.now();
            latenciesMs.push(finishedAt - startedAt);
          }

          delegateResults.push({
            id: spec.id,
            label: spec.label,
            stats: computeStats(latenciesMs),
            error: null,
          });
        } catch (err) {
          // A delegate that fails to load or run (e.g. Core ML unavailable
          // on this device) shouldn't lose the other delegate's results.
          delegateResults.push({
            id: spec.id,
            label: spec.label,
            stats: null,
            error: err instanceof Error ? err.message : String(err),
          });
        }
      }

      setResults(delegateResults);
      setStatus(delegateResults.some((result) => result.stats != null) ? 'done' : 'error');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus('error');
    }
  }, []);

  return { status, results, error, runBenchmark };
}

// Decodes/resizes to model input shape — only latency is measured, not parity with Python.
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
