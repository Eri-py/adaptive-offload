import type { ReactElement } from 'react';
import { StyleSheet, Text, View } from 'react-native';

export default function App(): ReactElement {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Mobile Inference Benchmark</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
  },
});
