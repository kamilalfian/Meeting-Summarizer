from dataclasses import dataclass
from pathlib import Path
from typing import List
from pydantic import BaseModel

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    learning_rate: float
    lr_scheduler_type: str
    metric_for_best_model: str
    greater_is_better: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    save_steps: int
    gradient_accumulation_steps: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path

class SummarizerSingleRequest(BaseModel):
    transcript: str

class SummarizerBatchRequest(BaseModel):
    features: List[SummarizerSingleRequest]

class SummarizerSingleResponse(BaseModel):
    result: str

class SummarizerBatchResponse(BaseModel):
    result: List[str]