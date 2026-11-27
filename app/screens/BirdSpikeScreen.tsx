import type { ReactElement } from 'react';
import { ActivityIndicator, Button, ScrollView, StyleSheet, Text, View } from 'react-native';
import { BIRD_PHOTO_COUNT, useBirdSpike, type SpikeResult } from '../hooks/useBirdSpike';

export default function BirdSpikeScreen(): ReactElement {
  const { status, results, error, runSpike } = useBirdSpike();

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Bird Classifier Spike</Text>
      <Text style={styles.subtitle}>
        MobileNetV3-Large · {BIRD_PHOTO_COUNT} photos · 2 layouts x 2 delegates
      </Text>

      <Button
        title={status === 'running' ? 'Running…' : 'Run Spike'}
        onPress={runSpike}
        disabled={status === 'running'}
      />

      {status === 'running' ? (
        <View style={styles.statusBlock}>
          <ActivityIndicator size="small" />
          <Text style={styles.statusText}>Running {results.length + 1} of 4…</Text>
        </View>
      ) : null}

      {status === 'idle' ? <Text style={styles.statusText}>Idle — tap to run</Text> : null}

      {error != null ? <Text style={[styles.statusText, styles.errorText]}>Error: {error}</Text> : null}

      {results.map((result) => (
        <SpikeCard key={result.key} result={result} />
      ))}
    </ScrollView>
  );
}

interface SpikeCardProps {
  result: SpikeResult;
}

function SpikeCard({ result }: SpikeCardProps): ReactElement {
  const { stats, agreement, worstProbDrift } = result;
  const agrees = agreement === BIRD_PHOTO_COUNT;

  return (
    <View style={styles.card}>
      <Text style={styles.cardHeading}>
        {result.layout} · {result.delegateLabel}
      </Text>

      {stats != null && agreement != null && worstProbDrift != null ? (
        <>
          <Text style={[styles.agreement, agrees ? styles.agreeGood : styles.agreeBad]}>
            {agreement}/{BIRD_PHOTO_COUNT} match PyTorch
          </Text>
          <Row label="Worst prob drift" value={worstProbDrift.toExponential(2)} />
          <Row label="Mean" value={`${stats.mean.toFixed(2)} ms`} />
          <Row label="Median" value={`${stats.median.toFixed(2)} ms`} />
          <Row label="Min" value={`${stats.min.toFixed(2)} ms`} />
          <Row label="Max" value={`${stats.max.toFixed(2)} ms`} />
          <Row label="Std dev" value={`${stats.stdev.toFixed(2)} ms`} />
        </>
      ) : (
        <Text style={[styles.statusText, styles.errorText]}>
          Error: {result.error ?? 'Unknown error'}
        </Text>
      )}
    </View>
  );
}

interface RowProps {
  label: string;
  value: string;
}

function Row({ label, value }: RowProps): ReactElement {
  return (
    <View style={styles.row}>
      <Text style={styles.label}>{label}</Text>
      <Text style={styles.value}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flexGrow: 1, padding: 20, paddingTop: 60, backgroundColor: '#fff' },
  title: { fontSize: 24, fontWeight: '700' },
  subtitle: { fontSize: 13, color: '#6e6e73', marginBottom: 20 },
  card: { backgroundColor: '#f2f2f7', borderRadius: 12, padding: 16, marginTop: 16 },
  cardHeading: {
    fontSize: 13,
    fontWeight: '600',
    textTransform: 'uppercase',
    color: '#6e6e73',
    marginBottom: 8,
  },
  agreement: { fontSize: 28, fontWeight: '800', marginBottom: 8 },
  agreeGood: { color: '#248a3d' },
  agreeBad: { color: '#d70015' },
  row: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 6 },
  label: { fontSize: 15 },
  value: { fontSize: 15, color: '#6e6e73' },
  statusBlock: { alignItems: 'center', marginTop: 12 },
  statusText: { textAlign: 'center', color: '#6e6e73', fontSize: 13, marginTop: 8 },
  errorText: { color: '#d70015' },
});
