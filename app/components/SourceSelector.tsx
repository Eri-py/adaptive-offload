import type { ReactElement } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import type { BundledSample } from '../services/samples';

export type SourceTab = 'sample' | 'camera' | 'library';

const TABS: { key: SourceTab; label: string }[] = [
  { key: 'sample', label: 'Samples' },
  { key: 'camera', label: 'Camera' },
  { key: 'library', label: 'Library' },
];

export interface SourceSelectorProps {
  activeTab: SourceTab;
  samples: BundledSample[];
  selectedSampleFile: string | null;
  disabled: boolean;
  onTabChange: (tab: SourceTab) => void;
  onSelectSample: (sample: BundledSample) => void;
  onPickCamera: () => void;
  onPickLibrary: () => void;
}

export default function SourceSelector({
  activeTab,
  samples,
  selectedSampleFile,
  disabled,
  onTabChange,
  onSelectSample,
  onPickCamera,
  onPickLibrary,
}: SourceSelectorProps): ReactElement {
  return (
    <View>
      <View style={styles.segment}>
        {TABS.map((tab) => (
          <Pressable
            key={tab.key}
            accessibilityRole="button"
            disabled={disabled}
            onPress={() => onTabChange(tab.key)}
            style={[styles.tab, activeTab === tab.key && styles.tabOn]}
          >
            <Text style={[styles.tabLabel, activeTab === tab.key && styles.tabLabelOn]}>{tab.label}</Text>
          </Pressable>
        ))}
      </View>

      {activeTab === 'sample' ? (
        <View style={styles.list}>
          {samples.map((sample) => (
            <Pressable
              key={sample.file}
              disabled={disabled}
              onPress={() => onSelectSample(sample)}
              style={[styles.sampleRow, selectedSampleFile === sample.file && styles.sampleRowOn]}
            >
              <Text style={styles.sampleName}>{sample.trueSpecies.name}</Text>
              <Text style={styles.sampleFile}>{sample.file}</Text>
            </Pressable>
          ))}
        </View>
      ) : (
        <Pressable
          disabled={disabled}
          onPress={activeTab === 'camera' ? onPickCamera : onPickLibrary}
          style={[styles.pickButton, disabled && styles.pickButtonDisabled]}
        >
          <Text style={styles.pickLabel}>{activeTab === 'camera' ? 'Take a photo' : 'Choose from library'}</Text>
        </Pressable>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  segment: { flexDirection: 'row', backgroundColor: '#e3e3e8', borderRadius: 9, padding: 2, marginBottom: 14 },
  tab: { flex: 1, alignItems: 'center', paddingVertical: 6, borderRadius: 7 },
  tabOn: { backgroundColor: '#fff' },
  tabLabel: { fontSize: 13, color: '#1c1c1e' },
  tabLabelOn: { fontWeight: '600' },
  list: { backgroundColor: '#f2f2f7', borderRadius: 12, marginBottom: 14, maxHeight: 220, overflow: 'hidden' },
  sampleRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 9, paddingHorizontal: 12 },
  sampleRowOn: { backgroundColor: '#d6e8ff' },
  sampleName: { fontSize: 14, color: '#1c1c1e' },
  sampleFile: { fontSize: 12, color: '#6e6e73' },
  pickButton: { backgroundColor: '#e3e3e8', borderRadius: 12, paddingVertical: 12, alignItems: 'center', marginBottom: 14 },
  pickButtonDisabled: { opacity: 0.5 },
  pickLabel: { fontSize: 15, fontWeight: '600', color: '#007aff' },
});
