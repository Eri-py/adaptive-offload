"""Response models for the benchmark frame endpoints."""

from pydantic import BaseModel


class BenchmarkFrame(BaseModel):
    """One frame the phone should benchmark, with the desktop figure to beat."""

    frame_id: str
    scene_complexity: float
    desktop_local_latency_ms: float


class BenchmarkFrameSet(BaseModel):
    """The frame set for one benchmark run, ordered least to most complex."""

    dataset: str
    frames: list[BenchmarkFrame]
