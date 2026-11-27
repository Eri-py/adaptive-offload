import { useCallback, useState } from 'react';
import { Asset } from 'expo-asset';
import { File } from 'expo-file-system';
import { decode as decodeJpeg } from 'jpeg-js';
import {
  loadTensorflowModel,
  type TfliteModel,
  type TensorflowModelDelegate,
} from 'react-native-fast-tflite';
import expectedRecords from '../assets/birds/expected.json';

const INPUT_SIZE = 224;
const INPUT_CHANNELS = 3;
const PIXELS = INPUT_SIZE * INPUT_SIZE;
const NUM_CLASSES = 200;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export type TensorLayout = 'NCHW' | 'NHWC';

interface ExpectedRecord {
  file: string;
  expectedLabel: number;
  expectedMaxProb: number;
}

const EXPECTED: ExpectedRecord[] = expectedRecords;

// Metro needs literal require() paths. Keyed by filename so order cannot drift from expected.json.
const BIRD_IMAGE_MODULES: Record<string, number> = {
  '1116.jpg': require('../assets/birds/1116.jpg'),
  '1176.jpg': require('../assets/birds/1176.jpg'),
  '7739.jpg': require('../assets/birds/7739.jpg'),
  '8996.jpg': require('../assets/birds/8996.jpg'),
  '2444.jpg': require('../assets/birds/2444.jpg'),
  '8481.jpg': require('../assets/birds/8481.jpg'),
  '1076.jpg': require('../assets/birds/1076.jpg'),
  '8693.jpg': require('../assets/birds/8693.jpg'),
  '5214.jpg': require('../assets/birds/5214.jpg'),
  '10116.jpg': require('../assets/birds/10116.jpg'),
  '5141.jpg': require('../assets/birds/5141.jpg'),
  '6254.jpg': require('../assets/birds/6254.jpg'),
  '9132.jpg': require('../assets/birds/9132.jpg'),
  '8243.jpg': require('../assets/birds/8243.jpg'),
  '11501.jpg': require('../assets/birds/11501.jpg'),
};

export const BIRD_PHOTO_COUNT = EXPECTED.length;

export type BirdSpikeStatus = 'idle' | 'running' | 'done' | 'error';

export interface SpikeStats {
  mean: number;
  median: number;
  min: number;
  max: number;
  stdev: number;
}

export interface SpikeResult {
  key: string;
  layout: TensorLayout;
  delegateLabel: string;
  stats: SpikeStats | null;
  agreement: number | null;
  worstProbDrift: number | null;
  error: string | null;
}

export interface UseBirdSpikeResult {
  status: BirdSpikeStatus;
  results: SpikeResult[];
  error: string | null;
  runSpike: () => Promise<void>;
}

interface ModelSpec {
  layout: TensorLayout;
  // Literal require() result; Metro resolves it at bundle time.
  source: number;
}

const MODEL_SPECS: ModelSpec[] = [
  { layout: 'NCHW', source: require('../assets/models/small.tflite') },
  { layout: 'NHWC', source: require('../assets/models/small-nhwc.tflite') },
];

const DELEGATE_SPECS: { label: string; delegates: TensorflowModelDelegate[] }[] = [
  { label: 'CPU', delegates: [] },
  { label: 'Core ML', delegates: ['core-ml'] },
];

export function useBirdSpike(): UseBirdSpikeResult {
  const [status, setStatus] = useState<BirdSpikeStatus>('idle');
  const [results, setResults] = useState<SpikeResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const runSpike = useCallback(async () => {
    setStatus('running');
    setError(null);
    setResults([]);

    try {
      // Decode once per layout is wasteful for a spike; decode each photo once, lay out per run.
      const photos = await Promise.all(EXPECTED.map((record) => loadPhotoRgba(record.file)));
      const collected: SpikeResult[] = [];

      for (const modelSpec of MODEL_SPECS) {
        for (const delegateSpec of DELEGATE_SPECS) {
          const result = await runCombination(modelSpec, delegateSpec, photos);
          collected.push(result);
          setResults([...collected]);
        }
      }

      setStatus(collected.some((result) => result.stats != null) ? 'done' : 'error');
    } catch (err) {
      setError(errorMessage(err));
      setStatus('error');
    }
  }, []);

  return { status, results, error, runSpike };
}

async function runCombination(
  modelSpec: ModelSpec,
  delegateSpec: { label: string; delegates: TensorflowModelDelegate[] },
  photos: Uint8Array[]
): Promise<SpikeResult> {
  const base = {
    key: `${modelSpec.layout}-${delegateSpec.label}`,
    layout: modelSpec.layout,
    delegateLabel: delegateSpec.label,
  };

  try {
    const model: TfliteModel = await loadTensorflowModel(modelSpec.source, delegateSpec.delegates);

    // Discarded warmup so Core ML compile / weight packing stays out of the statistics.
    const warmup = new Float32Array(PIXELS * INPUT_CHANNELS);
    await model.run([warmup.buffer as ArrayBuffer]);

    const latenciesMs: number[] = [];
    let agreement = 0;
    let worstProbDrift = 0;

    for (let i = 0; i < photos.length; i++) {
      const input = buildInputTensor(photos[i], modelSpec.layout);
      const startedAt = performance.now();
      // Cast is safe: freshly allocated, never a SharedArrayBuffer.
      const outputs = await model.run([input.buffer as ArrayBuffer]);
      latenciesMs.push(performance.now() - startedAt);

      const logits = new Float32Array(outputs[0]);
      if (logits.length !== NUM_CLASSES) {
        throw new Error(`Expected ${NUM_CLASSES} logits, got ${logits.length}`);
      }
      const { label, maxProb } = argmaxAndMaxProb(logits);
      if (label === EXPECTED[i].expectedLabel) {
        agreement += 1;
      }
      worstProbDrift = Math.max(worstProbDrift, Math.abs(maxProb - EXPECTED[i].expectedMaxProb));
    }

    return {
      ...base,
      stats: computeStats(latenciesMs),
      agreement,
      worstProbDrift,
      error: null,
    };
  } catch (err) {
    // A failing combination (e.g. Core ML rejecting the graph) must not lose the others.
    return { ...base, stats: null, agreement: null, worstProbDrift: null, error: errorMessage(err) };
  }
}

// Returns the decoded 224x224 RGBA pixels of one bundled photo.
async function loadPhotoRgba(fileName: string): Promise<Uint8Array> {
  const moduleId = BIRD_IMAGE_MODULES[fileName];
  if (moduleId == null) {
    throw new Error(`No bundled module registered for ${fileName}`);
  }
  const asset = Asset.fromModule(moduleId);
  await asset.downloadAsync();
  if (asset.localUri == null) {
    throw new Error(`Could not resolve a local file for ${fileName}`);
  }
  const jpegBytes = await new File(asset.localUri).arrayBuffer();
  const decoded = decodeJpeg(new Uint8Array(jpegBytes), { useTArray: true });
  if (decoded.width !== INPUT_SIZE || decoded.height !== INPUT_SIZE) {
    throw new Error(`${fileName} is ${decoded.width}x${decoded.height}, expected ${INPUT_SIZE}`);
  }
  // jpeg-js emits RGBA (4 bytes/pixel); the alpha byte is skipped below.
  return decoded.data;
}

const SOURCE_STRIDE = 4;

// ImageNet-normalizes RGBA bytes into the layout the model expects.
// NCHW is planar (all R, then all G, then all B); NHWC is interleaved.
function buildInputTensor(rgba: Uint8Array, layout: TensorLayout): Float32Array {
  const output = new Float32Array(PIXELS * INPUT_CHANNELS);

  for (let y = 0; y < INPUT_SIZE; y++) {
    for (let x = 0; x < INPUT_SIZE; x++) {
      const pixelIndex = y * INPUT_SIZE + x;
      const sourceIndex = pixelIndex * SOURCE_STRIDE;
      for (let channel = 0; channel < INPUT_CHANNELS; channel++) {
        const normalized = (rgba[sourceIndex + channel] / 255 - MEAN[channel]) / STD[channel];
        const destIndex =
          layout === 'NCHW' ? channel * PIXELS + pixelIndex : pixelIndex * INPUT_CHANNELS + channel;
        output[destIndex] = normalized;
      }
    }
  }

  return output;
}

// Numerically stable softmax: only the max probability is needed, which is 1 / sum(exp(l - max)).
function argmaxAndMaxProb(logits: Float32Array): { label: number; maxProb: number } {
  let label = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[label]) {
      label = i;
    }
  }
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    sum += Math.exp(logits[i] - logits[label]);
  }
  return { label, maxProb: 1 / sum };
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// Same shape/maths as computeStats in useBenchmark.ts (not exported there, and that file stays untouched).
function computeStats(valuesMs: number[]): SpikeStats {
  const sorted = [...valuesMs].sort((a, b) => a - b);
  const count = sorted.length;
  const mean = sorted.reduce((sum, value) => sum + value, 0) / count;
  const mid = Math.floor(count / 2);
  const median = count % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  const variance = sorted.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count;

  return { mean, median, min: sorted[0], max: sorted[count - 1], stdev: Math.sqrt(variance) };
}
