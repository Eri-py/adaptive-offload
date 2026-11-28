import { File, Paths } from 'expo-file-system';

export const DEFAULT_SERVER_URL = 'http://100.97.204.46:8000';

export type ServerUrlResult = {
  url: string;
  // True when the stored address was missing or unreadable, so the caller can tell the user.
  usedDefault: boolean;
};

function configFile(): File {
  return new File(Paths.document, 'server-config.json');
}

export async function readServerUrl(): Promise<ServerUrlResult> {
  try {
    const file = configFile();
    if (!file.exists) return { url: DEFAULT_SERVER_URL, usedDefault: true };
    const parsed: unknown = JSON.parse(await file.text());
    const url = (parsed as { baseUrl?: unknown } | null)?.baseUrl;
    if (typeof url === 'string' && url.trim() !== '') {
      return { url: url.trim(), usedDefault: false };
    }
  } catch (error) {
    console.warn('Server config unreadable, using default', error);
  }
  return { url: DEFAULT_SERVER_URL, usedDefault: true };
}

export async function writeServerUrl(url: string): Promise<void> {
  const trimmed = url.trim().replace(/\/+$/, '');
  if (!/^https?:\/\/\S+$/.test(trimmed)) throw new Error(`Not a valid server address: ${url}`);
  try {
    const file = configFile();
    if (!file.exists) file.create();
    file.write(JSON.stringify({ baseUrl: trimmed }));
  } catch (error) {
    throw new Error(`Could not save server address: ${String(error)}`);
  }
}
