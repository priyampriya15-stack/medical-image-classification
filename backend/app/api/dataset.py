from typing import Optional
from fastapi import APIRouter, Query
from app.schemas.dataset import DatasetInfoResponse
from app.services.dataset_service import dataset_service

router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.get("", response_model=DatasetInfoResponse)
@router.get("/", response_model=DatasetInfoResponse)
@router.get("/info", response_model=DatasetInfoResponse)
def get_dataset_info() -> DatasetInfoResponse:
    """
    Retrieve PneumoniaMNIST dataset structure, splits (train, validation, test),
    class balance, image dimensions, channels, and availability state directly from the loaded dataset archive.
    """
    return dataset_service.get_dataset_info()


@router.get("/samples", summary="Get real radiograph sample images from dataset")
def get_dataset_samples(
    split: Optional[str] = Query(None, description="Partition split: 'train', 'val', or 'test'"),
    page: int = Query(1, ge=1, description="Page index (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    count_per_class: Optional[int] = Query(None, description="Samples per class when split is not specified"),
):
    """
    Returns actual sample chest radiographs from the local PneumoniaMNIST archive as data URLs.
    Supports split-based pagination: GET /api/dataset/samples?split=test&page=1&page_size=20
    """
    return dataset_service.get_sample_items(
        count_per_class=count_per_class,
        split=split,
        page=page,
        page_size=page_size,
    )

