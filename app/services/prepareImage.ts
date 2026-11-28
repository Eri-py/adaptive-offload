import { File } from 'expo-file-system';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';

export const RESIZE_SHORT_SIDE = 232;
export const CROP_SIZE = 224;
export const JPEG_COMPRESS = 0.85;

export type PreparedImage = {
  jpegBytes: ArrayBuffer;
  byteCount: number;
  width: number;
  height: number;
};

// Mirrors the research pipeline: resize short side to 232, centre-crop 224, JPEG q85.
export async function prepareImage(uri: string): Promise<PreparedImage> {
  try {
    const source = await ImageManipulator.manipulate(uri).renderAsync();
    const isLandscape = source.width >= source.height;

    // Passing null for the long side lets the native module keep the aspect ratio.
    const resized = await ImageManipulator.manipulate(uri)
      .resize(isLandscape ? { width: null, height: RESIZE_SHORT_SIDE } : { width: RESIZE_SHORT_SIDE, height: null })
      .renderAsync();

    // Crop from the real resized size so native rounding cannot shift the origin.
    const cropped = await ImageManipulator.manipulate(resized)
      .crop({
        originX: Math.round((resized.width - CROP_SIZE) / 2),
        originY: Math.round((resized.height - CROP_SIZE) / 2),
        width: CROP_SIZE,
        height: CROP_SIZE,
      })
      .renderAsync();
    const saved = await cropped.saveAsync({ format: SaveFormat.JPEG, compress: JPEG_COMPRESS });

    // Raw bytes, not base64: base64 would inflate the payload by about a third.
    const jpegBytes = await new File(saved.uri).arrayBuffer();
    return { jpegBytes, byteCount: jpegBytes.byteLength, width: saved.width, height: saved.height };
  } catch (error) {
    throw new Error(`Could not prepare image: ${error instanceof Error ? error.message : String(error)}`);
  }
}
