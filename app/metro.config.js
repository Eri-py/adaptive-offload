const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// .tflite files load via require(), so Metro must treat .tflite as a bundleable asset extension.
config.resolver.assetExts.push('tflite');

module.exports = config;
