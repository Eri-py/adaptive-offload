import { Asset } from 'expo-asset';
import expectedRecords from '../assets/birds/expected.json';
import type { PhotoOutcome, TrueSpecies } from './pickPhoto';

interface ExpectedRecord {
  file: string;
  trueLabel: number;
  trueSpecies: string;
}

export interface BundledSample {
  file: string;
  trueSpecies: TrueSpecies;
}

const EXPECTED: ExpectedRecord[] = expectedRecords;

// Metro needs literal require() paths. Keyed by filename so order cannot drift from expected.json.
const SAMPLE_MODULES: Record<string, number> = {
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

export const BUNDLED_SAMPLES: BundledSample[] = EXPECTED.map((record) => ({
  file: record.file,
  trueSpecies: { label: record.trueLabel, name: record.trueSpecies },
}));

// Samples are already 224x224 at quality 85, so preparing them again is close to a no-op.
export async function loadSample(sample: BundledSample): Promise<PhotoOutcome> {
  try {
    const moduleId = SAMPLE_MODULES[sample.file];
    if (moduleId == null) {
      return { kind: 'error', message: `No bundled module registered for ${sample.file}` };
    }
    const asset = Asset.fromModule(moduleId);
    await asset.downloadAsync();
    if (asset.localUri == null) {
      return { kind: 'error', message: `Could not resolve a local file for ${sample.file}` };
    }
    return {
      kind: 'picked',
      photo: { uri: asset.localUri, source: 'sample', trueSpecies: sample.trueSpecies },
    };
  } catch (err) {
    return { kind: 'error', message: err instanceof Error ? err.message : String(err) };
  }
}
