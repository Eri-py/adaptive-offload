// eslint-disable-next-line @typescript-eslint/no-var-requires
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// react-native-fast-tflite ships models as .tflite files loaded via require();
// Metro needs to treat .tflite as a bundleable asset extension for that to
// resolve. See node_modules/react-native-fast-tflite/README.md, Installation
// step 2.
config.resolver.assetExts.push('tflite');

module.exports = config;
