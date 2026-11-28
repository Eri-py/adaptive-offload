import { useCallback, useEffect, useRef, useState } from 'react';
import { checkHealth } from '../services/offloadClient';

export type ServerHealthStatus = 'checking' | 'reachable' | 'unreachable';

export interface UseServerHealth {
  status: ServerHealthStatus;
  // Why the last check failed; null unless unreachable.
  reason: string | null;
  // Round trip of the last successful check; null otherwise.
  latencyMs: number | null;
  recheck: () => Promise<void>;
}

// Unreachable is a normal state here (the demo works offline), so nothing is thrown or alerted.
export function useServerHealth(): UseServerHealth {
  const [status, setStatus] = useState<ServerHealthStatus>('checking');
  const [reason, setReason] = useState<string | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const mounted = useRef(true);

  const recheck = useCallback(async (): Promise<void> => {
    setStatus('checking');
    let next: ServerHealthStatus;
    let nextReason: string | null = null;
    let nextLatency: number | null = null;
    try {
      const result = await checkHealth();
      if (result.ok) {
        next = 'reachable';
        nextLatency = result.value.roundTripMs;
      } else {
        next = 'unreachable';
        nextReason = result.failure.message;
      }
    } catch (error) {
      // checkHealth should not throw; guard anyway so a bug cannot leave "checking" forever.
      next = 'unreachable';
      nextReason = String(error);
    }
    if (!mounted.current) return;
    setStatus(next);
    setReason(nextReason);
    setLatencyMs(nextLatency);
  }, []);

  useEffect(() => {
    mounted.current = true;
    void recheck();
    return () => {
      mounted.current = false;
    };
  }, [recheck]);

  return { status, reason, latencyMs, recheck };
}
