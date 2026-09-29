"""Utility-based win/loss label for one (frame, condition) pair.

Consumes the four per-row latency/accuracy values from
`inference.apply_condition_overhead` and a lambda weight, and
scores each path as `utility = accuracy - lambda_value * (latency_ms / 1000)`
— see the spec's "Each (frame, condition) pair gets a computed win/loss
label" requirement. Pure function: no I/O, no database access.
"""

from __future__ import annotations

from common.models import Label

_MS_PER_SECOND = 1000.0


def _utility(accuracy: float, latency_ms: float, lambda_value: float) -> float:
    return accuracy - lambda_value * (latency_ms / _MS_PER_SECOND)


def compute_label(
    local_latency_ms: float,
    local_accuracy: float,
    offload_latency_ms: float,
    offload_accuracy: float,
    lambda_value: float,
) -> Label:
    """Return whichever path (`Label.LOCAL` or `Label.OFFLOAD`) has higher utility.

    On an exact utility tie, `Label.LOCAL` wins — an arbitrary but
    deterministic tie-break, consistent with this feature's other tie-break
    decisions (Task 7's remainder distribution). Local is picked as the
    default because it has no network dependency, making it the "safer"
    choice when the two paths are indistinguishable on the utility measure.
    """
    local_utility = _utility(local_accuracy, local_latency_ms, lambda_value)
    offload_utility = _utility(offload_accuracy, offload_latency_ms, lambda_value)

    if offload_utility > local_utility:
        return Label.OFFLOAD
    return Label.LOCAL
