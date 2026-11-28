import type { ReactElement } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import type { StyleProp, TextStyle } from 'react-native';
import type { PathState } from '../hooks/useScan';

export type Correctness = 'correct' | 'wrong' | 'unknown';

export interface PathResultCardProps {
  title: string;
  state: PathState;
  // 'unknown' when the photo has no ground truth: no verdict is shown.
  correctness: Correctness;
  isWinner: boolean;
  // What latencyMs minus modelMs is on this path, e.g. 'decode' or 'network + server overhead'.
  remainderLabel: string;
  // Shown while the path has not answered yet, e.g. 'sending 11.9 KB…'.
  waitingText: string;
}

function formatMs(ms: number): string {
  return ms < 10 ? `${ms.toFixed(1)} ms` : `${Math.round(ms)} ms`;
}

function tagFor(state: PathState, correctness: Correctness): { label: string; style: StyleProp<TextStyle> } {
  switch (state.status) {
    case 'idle':
      return { label: 'idle', style: styles.tag };
    case 'running':
      return { label: 'waiting', style: styles.tag };
    case 'failed':
      return { label: 'failed', style: styles.tagBad };
    case 'done':
      if (correctness === 'correct') return { label: 'correct', style: styles.tagGood };
      if (correctness === 'wrong') return { label: 'wrong', style: styles.tagBad };
      return { label: 'done', style: styles.tagGood };
  }
}

export default function PathResultCard({
  title,
  state,
  correctness,
  isWinner,
  remainderLabel,
  waitingText,
}: PathResultCardProps): ReactElement {
  const tag = tagFor(state, correctness);
  return (
    <View style={[styles.card, isWinner && styles.cardWin]}>
      <View style={styles.head}>
        <Text style={styles.title}>{title}</Text>
        <Text style={[styles.tagBase, tag.style]}>{tag.label}</Text>
      </View>

      {state.status === 'done' ? (
        <View>
          <Text
            style={[
              styles.species,
              correctness === 'correct' && styles.correct,
              correctness === 'wrong' && styles.wrong,
            ]}
          >
            {state.result.species}
          </Text>
          <Text style={styles.muted}>
            confidence {state.result.maxProbability.toFixed(2)} · margin {state.result.top2Margin.toFixed(2)}
          </Text>
          <Text style={styles.latency}>{formatMs(state.result.latencyMs)}</Text>
          <Text style={styles.small}>
            {formatMs(state.result.modelMs)} model ·{' '}
            {formatMs(Math.max(0, state.result.latencyMs - state.result.modelMs))} {remainderLabel}
          </Text>
        </View>
      ) : null}
      {state.status === 'running' ? <Text style={styles.note}>{waitingText}</Text> : null}
      {state.status === 'failed' ? <Text style={styles.note}>{state.reason}</Text> : null}
      {state.status === 'idle' ? <Text style={styles.note}>Not run yet.</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: '#f2f2f7', borderRadius: 12, padding: 14, marginBottom: 10, borderWidth: 2, borderColor: 'transparent' },
  cardWin: { borderColor: '#248a3d' },
  head: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  title: { fontSize: 11, fontWeight: '600', textTransform: 'uppercase', color: '#6e6e73', letterSpacing: 0.4 },
  tagBase: { fontSize: 10, fontWeight: '600', paddingHorizontal: 7, paddingVertical: 2, borderRadius: 20, overflow: 'hidden' },
  tag: { backgroundColor: '#e3e3e8', color: '#6e6e73' },
  tagGood: { backgroundColor: '#dcf2e1', color: '#248a3d' },
  tagBad: { backgroundColor: '#fde3e3', color: '#d70015' },
  species: { fontSize: 19, fontWeight: '700', color: '#1c1c1e' },
  correct: { color: '#248a3d' },
  wrong: { color: '#d70015' },
  muted: { fontSize: 13, color: '#6e6e73', marginTop: 2 },
  latency: { fontSize: 26, fontWeight: '800', marginTop: 8, color: '#1c1c1e' },
  small: { fontSize: 11, color: '#6e6e73' },
  note: { fontSize: 13, color: '#6e6e73', textAlign: 'center', paddingVertical: 10 },
});
