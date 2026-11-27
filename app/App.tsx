import { useState, type ReactElement } from 'react';
import { Button, StyleSheet, View } from 'react-native';
import BenchmarkScreen from './screens/BenchmarkScreen';
import BirdSpikeScreen from './screens/BirdSpikeScreen';

type ScreenId = 'benchmark' | 'bird-spike';

export default function App(): ReactElement {
  const [screen, setScreen] = useState<ScreenId>('benchmark');

  return (
    <View style={styles.root}>
      {screen === 'benchmark' ? <BenchmarkScreen /> : <BirdSpikeScreen />}
      <View style={styles.toggle}>
        <Button
          title={screen === 'benchmark' ? 'Switch to Bird Spike' : 'Switch to YOLO Benchmark'}
          onPress={() => setScreen(screen === 'benchmark' ? 'bird-spike' : 'benchmark')}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#fff' },
  toggle: { paddingBottom: 30, paddingTop: 8 },
});
