import * as ImagePicker from 'expo-image-picker';

export type PhotoSource = 'camera' | 'library' | 'sample';

// Known only for bundled samples; camera and library photos carry `null`, never a default.
export interface TrueSpecies {
  label: number;
  name: string;
}

export interface ChosenPhoto {
  // Local file URI, directly usable by expo-image-manipulator.
  uri: string;
  source: PhotoSource;
  trueSpecies: TrueSpecies | null;
}

export type PhotoOutcome =
  | { kind: 'picked'; photo: ChosenPhoto }
  // Not an error: the UI should simply return to idle.
  | { kind: 'cancelled' }
  | { kind: 'denied'; source: PhotoSource; canAskAgain: boolean }
  | { kind: 'error'; message: string };

export function pickFromCamera(): Promise<PhotoOutcome> {
  return pick('camera', ImagePicker.requestCameraPermissionsAsync, ImagePicker.launchCameraAsync);
}

export function pickFromLibrary(): Promise<PhotoOutcome> {
  // The system picker needs no library permission on recent iOS, but the request keeps older paths honest.
  return pick(
    'library',
    ImagePicker.requestMediaLibraryPermissionsAsync,
    ImagePicker.launchImageLibraryAsync
  );
}

async function pick(
  source: 'camera' | 'library',
  requestPermission: () => Promise<ImagePicker.PermissionResponse>,
  launch: (options?: ImagePicker.ImagePickerOptions) => Promise<ImagePicker.ImagePickerResult>
): Promise<PhotoOutcome> {
  try {
    const permission = await requestPermission();
    if (!permission.granted) {
      return { kind: 'denied', source, canAskAgain: permission.canAskAgain };
    }

    // One image, no editing: the preparation step does its own centre crop.
    const result = await launch({ mediaTypes: ['images'], allowsMultipleSelection: false });
    if (result.canceled) {
      return { kind: 'cancelled' };
    }
    const asset = result.assets[0];
    if (asset == null) {
      return { kind: 'error', message: 'The picker returned no photo' };
    }
    return { kind: 'picked', photo: { uri: asset.uri, source, trueSpecies: null } };
  } catch (err) {
    return { kind: 'error', message: err instanceof Error ? err.message : String(err) };
  }
}
