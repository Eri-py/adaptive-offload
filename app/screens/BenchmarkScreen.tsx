import type { ReactElement } from 'react';
import { ActivityIndicator, Button, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useBenchmark } from '../hooks/useBenchmark';

// Describes the fixed bundled test set (Task 1's 15 COCO images + exported
// model) -- not derived from useBenchmark, which only reports run results.
const BUNDLED_IMAGE_COUNT = 15;
const RUNS_PER_IMAGE = 1;
const MODEL_NAME = 'yolov8n.tflite';

export default function BenchmarkScreen(): ReactElement {
  const { status, results, error, runBenchmark } = useBenchmark();

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Local Inference Bench</Text>
      <Text style={styles.subtitle}>YOLOv8n · TFLite · on-device</Text>

      <View style={styles.card}>
        <Text style={styles.cardHeading}>Bundled test set</Text>
        <View style={styles.row}>
          <Text style={styles.label}>Images</Text>
          <Text style={styles.value}>{BUNDLED_IMAGE_COUNT}</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Runs per image</Text>
          <Text style={styles.value}>{RUNS_PER_IMAGE}</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Model</Text>
          <Text style={styles.value}>{MODEL_NAME}</Text>
        </View>
      </View>

      <Button
        title={status === 'running' ? 'Running…' : 'Run Benchmark'}
        onPress={runBenchmark}
        disabled={status === 'running'}
      />

      {status === 'running' ? (
        <View style={styles.statusBlock}>
          <ActivityIndicator size="small" />
          <Text style={styles.statusText}>Running benchmark…</Text>
        </View>
      ) : null}

      {status === 'idle' ? (
        <Text style={styles.statusText}>Idle — tap to run {BUNDLED_IMAGE_COUNT} inferences</Text>
      ) : null}

      {status === 'error' && results == null ? (
        <Text style={[styles.statusText, styles.errorText]}>
          Error: {error ?? 'Unknown error'}
        </Text>
      ) : null}

      {results != null
        ? results.map((delegateResult) => (
            <View key={delegateResult.id} style={styles.card}>
              <Text style={styles.cardHeading}>Delegate: {delegateResult.label}</Text>
              {delegateResult.stats != null ? (
                <>
                  <View style={styles.row}>
                    <Text style={styles.label}>Mean</Text>
                    <Text style={styles.value}>{delegateResult.stats.mean.toFixed(2)} ms</Text>
                  </View>
                  <View style={styles.row}>
                    <Text style={styles.label}>Median</Text>
                    <Text style={styles.value}>{delegateResult.stats.median.toFixed(2)} ms</Text>
                  </View>
                  <View style={styles.row}>
                    <Text style={styles.label}>Min</Text>
                    <Text style={styles.value}>{delegateResult.stats.min.toFixed(2)} ms</Text>
                  </View>
                  <View style={styles.row}>
                    <Text style={styles.label}>Max</Text>
                    <Text style={styles.value}>{delegateResult.stats.max.toFixed(2)} ms</Text>
                  </View>
                  <View style={styles.row}>
                    <Text style={styles.label}>Std dev</Text>
                    <Text style={styles.value}>{delegateResult.stats.stdev.toFixed(2)} ms</Text>
                  </View>
                </>
              ) : (
                <Text style={[styles.statusText, styles.errorText]}>
                  Error: {delegateResult.error ?? 'Unknown error'}
                </Text>
              )}
            </View>
          ))
        : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 20,
    paddingTop: 60,
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
  },
  subtitle: {
    fontSize: 13,
    color: '#6e6e73',
    marginBottom: 20,
  },
  card: {
    backgroundColor: '#f2f2f7',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  cardHeading: {
    fontSize: 13,
    fontWeight: '600',
    textTransform: 'uppercase',
    color: '#6e6e73',
    marginBottom: 8,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 6,
  },
  label: {
    fontSize: 15,
  },
  value: {
    fontSize: 15,
    color: '#6e6e73',
  },
  statusBlock: {
    alignItems: 'center',
    marginTop: 12,
  },
  statusText: {
    textAlign: 'center',
    color: '#6e6e73',
    fontSize: 13,
    marginTop: 8,
  },
  errorText: {
    color: '#d70015',
  },
});
