/**
 * Minimal ambient declaration for `jpeg-js` (installed version: 0.4.4).
 *
 * The `@types/jpeg-js` package on npm (0.3.x) documents the *old* 0.3 API,
 * where `decode`'s second argument is a boolean. The real 0.4.4 runtime API
 * (confirmed against `node_modules/jpeg-js/README.md`) takes an options
 * object instead (`{ useTArray, formatAsRGBA, ... }`), so the DefinitelyTyped
 * package was intentionally not installed — it would type-check against an
 * API this version doesn't have. Only the `decode` call shape this hook
 * actually uses is declared here.
 */
declare module 'jpeg-js' {
  export interface JpegDecodeOptions {
    /** Transform alternate colorspaces like YCbCr. Default: respects file metadata. */
    colorTransform?: boolean;
    /** Decode pixels into a typed Uint8Array instead of a Buffer. Default: false. */
    useTArray?: boolean;
    /** Decode pixels into RGBA (4 channels) vs. RGB (3 channels). Default: true. */
    formatAsRGBA?: boolean;
    /** Be more tolerant of technically invalid JPEGs. Default: true. */
    tolerantDecoding?: boolean;
    /** Maximum resolution (in megapixels) this call will attempt to decode. Default: 100. */
    maxResolutionInMP?: number;
    /** Maximum memory (in MiB) this call will allocate while decoding. Default: 512. */
    maxMemoryUsageInMB?: number;
  }

  export interface RawImageData<TData extends Uint8Array = Uint8Array> {
    width: number;
    height: number;
    data: TData;
  }

  export function decode(
    jpegData: Uint8Array | ArrayBuffer,
    options?: JpegDecodeOptions
  ): RawImageData;
}
