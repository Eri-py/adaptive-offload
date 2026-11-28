import type { ReactElement } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import type { AccuracyWinner, LatencyWinner, ScanSummary } from '../hooks/useScan';

export interface ScanSummaryCardProps {
  summary: ScanSummary;
  // Display names for the two paths, so the card does not hard-code them.
  localName: string;
  offloadName: string;
  payloadBytes: number | null;
  // Present only when both paths answered; used for the latency gap.
  localLatencyMs: number | null;
  offloadLatencyMs: number | null;
}

function latencyText(
  winner: LatencyWinner,
  localName: string,
  offloadName: string,
  gapMs: number | null
): string {
  if (winner === 'unavailable') return 'unavailable (a path failed)';
  const name = winner === 'local' ? localName : offloadName;
  return gapMs === null ? name : `${name} (${Math.round(gapMs)} ms faster)`;
}

function accuracyText(winner: AccuracyWinner, localName: string, offloadName: string): string {
  switch (winner) {
    case 'local':
      return localName;
    case 'offload':
      return offloadName;
    case 'both':
      return 'both correct';
    case 'neither':
      return 'neither correct';
    case 'unknown':
      return 'unknown (no ground truth)';
    case 'unavailable':
      return 'unavailable (a path failed)';
  }
}

export default function ScanSummaryCard({
  summary,
  localName,
  offloadName,
  payloadBytes,
  localLatencyMs,
  offloadLatencyMs,
}: ScanSummaryCardProps): ReactElement {
  const gapMs =
    localLatencyMs !== null && offloadLatencyMs !== null ? Math.abs(localLatencyMs - offloadLatencyMs) : null;
  return (
    <View style={styles.card}>
      <Text style={styles.title}>This scan</Text>
      <View style={styles.row}>
        <Text style={styles.label}>Faster</Text>
        <Text style={styles.value}>{latencyText(summary.latencyWinner, localName, offloadName, gapMs)}</Text>
      </View>
      <View style={styles.row}>
        <Text style={styles.label}>More accurate</Text>
        <Text style={styles.value}>{accuracyText(summary.accuracyWinner, localName, offloadName)}</Text>
      </View>
      {payloadBytes !== null ? (
        <View style={styles.row}>
          <Text style={styles.label}>Payload</Text>
          <Text style={styles.value}>{payloadBytes.toLocaleString('en-US')} bytes</Text>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: '#f2f2f7', borderRadius: 12, padding: 14, marginBottom: 10 },
  title: { fontSize: 11, fontWeight: '600', textTransform: 'uppercase', color: '#6e6e73', letterSpacing: 0.4, marginBottom: 8 },
  row: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 3 },
  label: { fontSize: 13, color: '#1c1c1e' },
  value: { fontSize: 13, color: '#6e6e73', flexShrink: 1, textAlign: 'right', marginLeft: 12 },
});
