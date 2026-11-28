import { useCallback, useEffect, useState } from 'react';
import type { ReactElement } from 'react';
import { Image, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import PathResultCard from '../components/PathResultCard';
import type { Correctness } from '../components/PathResultCard';
import ScanSummaryCard from '../components/ScanSummaryCard';
import SourceSelector from '../components/SourceSelector';
import type { SourceTab } from '../components/SourceSelector';
import { useScan } from '../hooks/useScan';
import type { PathName, PathState, ScanSummary } from '../hooks/useScan';
import { useServerHealth } from '../hooks/useServerHealth';
import { pickFromCamera, pickFromLibrary } from '../services/pickPhoto';
import type { ChosenPhoto, PhotoOutcome, PhotoSource } from '../services/pickPhoto';
import { BUNDLED_SAMPLES, loadSample } from '../services/samples';
import type { BundledSample } from '../services/samples';
import { readServerUrl, writeServerUrl } from '../services/serverConfig';

const SOURCE_LABEL: Record<PhotoSource, string> = {
  camera: 'Camera',
  library: 'Photo library',
  sample: 'Samples',
};

function correctnessOf(state: PathState, photo: ChosenPhoto | null): Correctness {
  if (state.status !== 'done' || photo?.trueSpecies == null) return 'unknown';
  return state.result.classIndex === photo.trueSpecies.label ? 'correct' : 'wrong';
}

// A single accuracy winner takes the outline; otherwise the latency winner; otherwise whichever path answered.
function outlinedPath(summary: ScanSummary | null, local: PathState, offload: PathState): PathName | null {
  if (summary === null) return null;
  if (summary.accuracyWinner === 'local' || summary.accuracyWinner === 'offload') return summary.accuracyWinner;
  if (summary.latencyWinner !== 'unavailable') return summary.latencyWinner;
  if (local.status === 'done') return 'local';
  if (offload.status === 'done') return 'offload';
  return null;
}

// Cancelled and picked produce no message: cancelling is not an error, and a pick clears any old one.
function noticeFor(outcome: PhotoOutcome): string | null {
  switch (outcome.kind) {
    case 'cancelled':
    case 'picked':
      return null;
    case 'denied': {
      const hint = outcome.canAskAgain ? 'Try again to grant it.' : 'Enable it in Settings.';
      return `${SOURCE_LABEL[outcome.source]} access was denied. ${hint} The other sources still work.`;
    }
    case 'error':
      return `Could not load the photo: ${outcome.message}`;
  }
}

export default function ScanScreen(): ReactElement {
  const scan = useScan();
  const health = useServerHealth();
  const { reset, startScan } = scan;
  const { recheck } = health;
  const [tab, setTab] = useState<SourceTab>('sample');
  const [chosen, setChosen] = useState<ChosenPhoto | null>(null);
  const [selectedSampleFile, setSelectedSampleFile] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [addressDraft, setAddressDraft] = useState('');
  const [addressMessage, setAddressMessage] = useState<string | null>(null);
  const [usingDefault, setUsingDefault] = useState(false);

  useEffect(() => {
    let active = true;
    readServerUrl()
      .then((result) => {
        if (!active) return;
        setAddressDraft(result.url);
        setUsingDefault(result.usedDefault);
      })
      .catch((error: unknown) => {
        if (active) setAddressMessage(`Could not read the saved address: ${String(error)}`);
      });
    return () => {
      active = false;
    };
  }, []);

  // A failed offload can mean the server just went away; refresh the tag beside it.
  const offloadFailed = scan.phase === 'settled' && scan.offload.status === 'failed';
  useEffect(() => {
    if (offloadFailed) void recheck();
  }, [offloadFailed, recheck]);

  const handleOutcome = useCallback(
    (outcome: PhotoOutcome, sampleFile: string | null): void => {
      setNotice(noticeFor(outcome));
      // Cancelled keeps whatever was chosen before; only a real pick replaces it.
      if (outcome.kind === 'picked') {
        setChosen(outcome.photo);
        setSelectedSampleFile(sampleFile);
        reset();
      }
    },
    [reset]
  );

  const handleSample = useCallback(
    async (sample: BundledSample): Promise<void> => handleOutcome(await loadSample(sample), sample.file),
    [handleOutcome]
  );
  const handleCamera = useCallback(
    async (): Promise<void> => handleOutcome(await pickFromCamera(), null),
    [handleOutcome]
  );
  const handleLibrary = useCallback(
    async (): Promise<void> => handleOutcome(await pickFromLibrary(), null),
    [handleOutcome]
  );

  const handleSaveAddress = useCallback(async (): Promise<void> => {
    try {
      await writeServerUrl(addressDraft);
      setAddressMessage('Saved.');
      setUsingDefault(false);
      await recheck();
    } catch (error) {
      setAddressMessage(error instanceof Error ? error.message : String(error));
    }
  }, [addressDraft, recheck]);

  const busy = scan.phase === 'preparing' || scan.phase === 'running';
  const settled = scan.phase === 'settled';
  const finished = settled || scan.phase === 'prepare-failed';
  const showPaths = scan.phase === 'running' || settled;
  const outlined = outlinedPath(scan.summary, scan.local, scan.offload);
  const payloadText = scan.byteCount === null ? '' : `${(scan.byteCount / 1024).toFixed(1)} KB`;
  let buttonLabel = 'Scan';
  if (scan.phase === 'preparing') buttonLabel = 'Preparing…';
  else if (scan.phase === 'running') buttonLabel = 'Scanning…';
  else if (finished) buttonLabel = 'Scan again';
  const buttonOff = chosen === null || busy;

  return (
    <ScrollView style={styles.root} contentContainerStyle={styles.content}>
      <Text style={styles.heading}>Scan</Text>
      <Text style={styles.sub}>MobileNetV3 on device · ConvNeXt on server</Text>

      <SourceSelector
        activeTab={tab}
        samples={BUNDLED_SAMPLES}
        selectedSampleFile={selectedSampleFile}
        disabled={busy}
        onTabChange={setTab}
        onSelectSample={handleSample}
        onPickCamera={handleCamera}
        onPickLibrary={handleLibrary}
      />

      {scan.modelError !== null ? (
        <View style={styles.error}>
          <Text style={styles.errorTitle}>On-device model unavailable</Text>
          <Text style={styles.errorText}>{scan.modelError}</Text>
        </View>
      ) : null}

      {notice !== null ? (
        <View style={styles.error}>
          <Text style={styles.errorText}>{notice}</Text>
        </View>
      ) : null}

      {chosen === null ? (
        <View style={[styles.photo, styles.photoEmpty]}>
          <Text style={styles.photoEmptyText}>Pick a sample bird photo,{'\n'}or take one with the camera</Text>
        </View>
      ) : (
        <View style={styles.photo}>
          <Image source={{ uri: chosen.uri }} style={styles.photoImage} resizeMode="cover" />
          {scan.prepared !== null ? (
            <View style={styles.pills}>
              <Text style={styles.pill}>
                {scan.prepared.width} × {scan.prepared.height}
              </Text>
              <Text style={styles.pill}>{payloadText}</Text>
            </View>
          ) : null}
        </View>
      )}
      <Text style={styles.meta}>{scan.prepared !== null ? 'cropped · JPEG q85' : ' '}</Text>

      <Pressable
        accessibilityRole="button"
        disabled={buttonOff}
        onPress={() => {
          if (chosen !== null) void startScan(chosen);
        }}
        style={[styles.button, finished && styles.buttonSecondary, buttonOff && styles.buttonDisabled]}
      >
        <Text style={[styles.buttonLabel, finished && styles.buttonLabelSecondary]}>{buttonLabel}</Text>
      </Pressable>

      {scan.phase === 'prepare-failed' ? (
        <View style={styles.error}>
          <Text style={styles.errorTitle}>Could not prepare the photo</Text>
          <Text style={styles.errorText}>{scan.prepareError}</Text>
        </View>
      ) : null}

      {showPaths && chosen !== null ? (
        <View>
          {settled ? (
            <View style={styles.truth}>
              <Text style={styles.truthLabel}>Ground truth</Text>
              <Text style={styles.truthText}>
                {chosen.trueSpecies !== null
                  ? chosen.trueSpecies.name
                  : 'Unknown for this photo, so no correct/wrong verdict'}
              </Text>
            </View>
          ) : null}

          {scan.offload.status === 'failed' ? (
            <View style={styles.error}>
              <Text style={styles.errorTitle}>Offload failed</Text>
              <Text style={styles.errorText}>{scan.offload.reason} The on-device result is unaffected.</Text>
            </View>
          ) : null}

          <PathResultCard
            title="On device"
            state={scan.local}
            correctness={correctnessOf(scan.local, chosen)}
            isWinner={outlined === 'local'}
            remainderLabel="decode"
            waitingText="running…"
          />
          <PathResultCard
            title="Server"
            state={scan.offload}
            correctness={correctnessOf(scan.offload, chosen)}
            isWinner={outlined === 'offload'}
            remainderLabel="network + server overhead"
            waitingText={`sending ${payloadText}…`}
          />

          {scan.summary !== null ? (
            <ScanSummaryCard
              summary={scan.summary}
              localName="On device"
              offloadName="Server"
              payloadBytes={scan.byteCount}
              localLatencyMs={scan.local.status === 'done' ? scan.local.result.latencyMs : null}
              offloadLatencyMs={scan.offload.status === 'done' ? scan.offload.result.latencyMs : null}
            />
          ) : null}
        </View>
      ) : null}

      <View style={styles.serverCard}>
        <View style={styles.serverHead}>
          <Text style={styles.serverTitle}>Server</Text>
          {health.latencyMs !== null ? (
            <Text style={styles.serverLatency}>{Math.round(health.latencyMs)} ms</Text>
          ) : null}
          <Text
            style={[
              styles.tag,
              health.status === 'reachable' && styles.tagGood,
              health.status === 'unreachable' && styles.tagBad,
            ]}
          >
            {health.status}
          </Text>
        </View>
        {health.reason !== null ? <Text style={styles.serverReason}>{health.reason}</Text> : null}
        <TextInput
          value={addressDraft}
          onChangeText={setAddressDraft}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          editable={!busy}
          style={styles.addressInput}
        />
        <Pressable disabled={busy} onPress={handleSaveAddress} style={styles.saveButton}>
          <Text style={styles.saveLabel}>Save and check</Text>
        </Pressable>
        {usingDefault ? <Text style={styles.serverReason}>No saved address; using the default.</Text> : null}
        {addressMessage !== null ? <Text style={styles.serverReason}>{addressMessage}</Text> : null}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#fff' },
  content: { paddingTop: 60, paddingHorizontal: 16, paddingBottom: 32 },
  heading: { fontSize: 24, fontWeight: '700', color: '#1c1c1e' },
  sub: { fontSize: 13, color: '#6e6e73', marginTop: 2, marginBottom: 16 },
  photo: { width: '100%', aspectRatio: 1, borderRadius: 12, marginBottom: 6, overflow: 'hidden', backgroundColor: '#f2f2f7' },
  photoEmpty: { alignItems: 'center', justifyContent: 'center', padding: 20 },
  photoEmptyText: { color: '#6e6e73', fontSize: 14, textAlign: 'center' },
  photoImage: { width: '100%', height: '100%' },
  pills: { position: 'absolute', left: 8, right: 8, bottom: 8, flexDirection: 'row', justifyContent: 'space-between' },
  pill: { backgroundColor: 'rgba(0,0,0,0.6)', color: '#fff', fontSize: 11, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 20, overflow: 'hidden' },
  meta: { fontSize: 11, color: '#6e6e73', textAlign: 'right', marginBottom: 14 },
  button: { backgroundColor: '#007aff', borderRadius: 12, paddingVertical: 14, alignItems: 'center', marginBottom: 14 },
  buttonSecondary: { backgroundColor: '#e3e3e8' },
  buttonDisabled: { opacity: 0.4 },
  buttonLabel: { color: '#fff', fontSize: 17, fontWeight: '600' },
  buttonLabelSecondary: { color: '#007aff' },
  error: { backgroundColor: '#fde3e3', borderRadius: 12, padding: 12, marginBottom: 10 },
  errorTitle: { color: '#d70015', fontSize: 13, fontWeight: '700', marginBottom: 2 },
  errorText: { color: '#d70015', fontSize: 13 },
  truth: { backgroundColor: '#fff8e8', borderRadius: 12, paddingVertical: 10, paddingHorizontal: 14, marginBottom: 10 },
  truthLabel: { fontSize: 11, fontWeight: '700', textTransform: 'uppercase', color: '#b25000', letterSpacing: 0.4, marginBottom: 2 },
  truthText: { fontSize: 13, color: '#1c1c1e' },
  serverCard: { backgroundColor: '#f2f2f7', borderRadius: 12, padding: 14, marginTop: 4 },
  serverHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  serverTitle: { fontSize: 11, fontWeight: '600', textTransform: 'uppercase', color: '#6e6e73', letterSpacing: 0.4 },
  serverLatency: { flex: 1, fontSize: 12, color: '#6e6e73', marginLeft: 8 },
  tag: { fontSize: 10, fontWeight: '600', paddingHorizontal: 7, paddingVertical: 2, borderRadius: 20, overflow: 'hidden', backgroundColor: '#e3e3e8', color: '#6e6e73' },
  tagGood: { backgroundColor: '#dcf2e1', color: '#248a3d' },
  tagBad: { backgroundColor: '#fde3e3', color: '#d70015' },
  serverReason: { fontSize: 12, color: '#6e6e73', marginTop: 6 },
  addressInput: { backgroundColor: '#fff', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 8, fontSize: 14, color: '#1c1c1e', marginTop: 4 },
  saveButton: { alignItems: 'center', paddingVertical: 10 },
  saveLabel: { color: '#007aff', fontSize: 15, fontWeight: '600' },
});
