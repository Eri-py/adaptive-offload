"""Classify endpoint: raw JPEG body in, species prediction out."""

import time
from typing import Annotated

import torch
from fastapi import APIRouter, Depends, HTTPException
from torch import nn

from api.contracts.classification import ClassifyResponse, SpeciesProbability
from api.dependencies.providers import (
    get_device,
    get_raw_body,
    get_request_logger,
    get_request_start,
    get_required_model,
    get_species,
)
from api.services.classification.classify import classify
from api.services.classification.preprocess import ImageDecodeError, preprocess_image
from api.services.classification.request_log import RequestLogger

router = APIRouter()


@router.post("/classify")
def classify_image(
    started_at: Annotated[float, Depends(get_request_start)],
    body: Annotated[bytes, Depends(get_raw_body)],
    model: Annotated[nn.Module, Depends(get_required_model)],
    species: Annotated[list[str], Depends(get_species)],
    device: Annotated[torch.device, Depends(get_device)],
    log_request: Annotated[RequestLogger, Depends(get_request_logger)],
) -> ClassifyResponse:
    try:
        preprocessed = preprocess_image(body)
    except ImageDecodeError as error:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {error}") from error
    result = classify(model, species, preprocessed.tensor, device)
    response = ClassifyResponse(
        species=result.top1.name,
        class_index=result.top1.index,
        max_probability=result.max_probability,
        margin=result.margin,
        top3=[
            SpeciesProbability(index=p.index, name=p.name, probability=p.probability)
            for p in result.top3
        ],
        compute_ms=result.compute_ms,
        geometric_transform_applied=preprocessed.geometric_transform_applied,
    )
    log_request(
        result,
        payload_bytes=len(body),
        was_transformed=preprocessed.geometric_transform_applied,
        total_ms=(time.perf_counter() - started_at) * 1000,
    )
    return response
