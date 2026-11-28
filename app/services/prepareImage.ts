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

export type ResizeTarget = { width: number | null; height: number | null };
export type CropOrigin = { originX: number; originY: number };

// Passing null for the long side lets the native module keep the aspect ratio.
export function shortSideResize(width: number, height: number): ResizeTarget {
  return width >= height ? { width: null, height: RESIZE_SHORT_SIDE } : { width: RESIZE_SHORT_SIDE, height: null };
}

// Clamped at 0 so a mis-reported orientation cannot yield a negative origin.
export function cropOrigin(resizedWidth: number, resizedHeight: number): CropOrigin {
  return {
    originX: Math.max(0, Math.round((resizedWidth - CROP_SIZE) / 2)),
    originY: Math.max(0, Math.round((resizedHeight - CROP_SIZE) / 2)),
  };
}

// An exactly 224x224 source is already a research crop; re-transforming it drifts the tensor.
export function isAlreadyPrepared(width: number, height: number): boolean {
  return width === CROP_SIZE && height === CROP_SIZE;
}

// Mirrors the research pipeline: resize short side to 232, centre-crop 224, JPEG q85.
export async function prepareImage(uri: string): Promise<PreparedImage> {
  try {
    const source = await ImageManipulator.manipulate(uri).renderAsync();

    if (isAlreadyPrepared(source.width, source.height)) {
      // Mirrors the server, which also leaves a 224x224 input untransformed.
      const original = await new File(uri).arrayBuffer();
      return { jpegBytes: original, byteCount: original.byteLength, width: source.width, height: source.height };
    }

    // Chain from the rendered ref so the source file is decoded only once.
    const resized = await ImageManipulator.manipulate(source)
      .resize(shortSideResize(source.width, source.height))
      .renderAsync();
    if (resized.width < CROP_SIZE || resized.height < CROP_SIZE) {
      throw new Error(`resized image is ${resized.width}x${resized.height}, smaller than ${CROP_SIZE}`);
    }

    // Crop from the real resized size so native rounding cannot shift the origin.
    const cropped = await ImageManipulator.manipulate(resized)
      .crop({ ...cropOrigin(resized.width, resized.height), width: CROP_SIZE, height: CROP_SIZE })
      .renderAsync();
    const saved = await cropped.saveAsync({ format: SaveFormat.JPEG, compress: JPEG_COMPRESS });

    // Raw bytes, not base64: base64 would inflate the payload by about a third.
    const jpegBytes = await new File(saved.uri).arrayBuffer();
    return { jpegBytes, byteCount: jpegBytes.byteLength, width: saved.width, height: saved.height };
  } catch (error) {
    throw new Error(`Could not prepare image: ${error instanceof Error ? error.message : String(error)}`);
  }
}
