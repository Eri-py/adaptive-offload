"""Tests for `birds.router.conditions`. Server compute and payload are required inputs with no
default, so these tests pass their own synthetic values; they are not measurements."""

from __future__ import annotations

from birds.router.conditions import (
    LOCAL_SWEEP_MS,
    NETWORK_CONDITIONS,
    OFFLOAD_SWEEP_MS,
    SMALL_DESKTOP_CPU_MS,
    named_offload_costs,
    offload_cost_ms,
    transfer_ms,
)

SYNTHETIC_PAYLOAD_KB = 20.0
SYNTHETIC_SERVER_MS = 10.0


def test_transfer_converts_kilobytes_and_megabits_to_milliseconds() -> None:
    assert transfer_ms(20.0, 10.0) == 16.0


def test_offload_cost_sums_server_transfer_and_round_trip() -> None:
    cost = offload_cost_ms(
        payload_kb=20.0, bandwidth_mbps=10.0, round_trip_ms=60.0, server_compute_ms=7.0
    )
    assert cost == 7.0 + 16.0 + 60.0


def test_named_conditions_are_distinct_and_ordered() -> None:
    costs = named_offload_costs(
        payload_kb=SYNTHETIC_PAYLOAD_KB, server_compute_ms=SYNTHETIC_SERVER_MS
    )
    assert list(costs) == ["good", "medium", "poor"] == [c.name for c in NETWORK_CONDITIONS]
    assert costs["good"] < costs["medium"] < costs["poor"]


def test_sweeps_cover_the_required_range_and_mark_the_real_measurement() -> None:
    assert LOCAL_SWEEP_MS.min() == 1.0
    assert LOCAL_SWEEP_MS.max() == 150.0
    assert LOCAL_SWEEP_MS.min() <= SMALL_DESKTOP_CPU_MS <= LOCAL_SWEEP_MS.max()
    assert OFFLOAD_SWEEP_MS.min() > 0
