import typed_settings as ts
from pathlib import Path
from enum import Enum


class Mode(Enum):
    TrainLightning = 0
    RunFilter = 1


class Split(Enum):
    train = "train"
    test = "test"


@ts.settings
class Train:
    split: Split = Split.train
    learning_rate: float = 0.0005
    batch_size: int = 32
    max_epochs: int = 100
    gpus: int = 1


@ts.settings
class Test:
    split: Split = Split.test


@ts.settings
class Data:
    kernel_len: int = 5
    infeasbile_space_weight: int = 10000
    sample_rate: int = 60
    num_beacons: int = 4


@ts.settings
class Filter:
    num_particles: int = 1000
    update_rate: int = 60
    init_std: float = 0.01
    odom_std: float = 0.1
    history_length: int = 5
    allow_ble_update: bool = True
    allow_reinit: bool = True
    reinit_percent: float = 0.9
    infeasible_space_weight: int = 10000


@ts.settings
class Config:
    dataset: str
    building: str
    mode: Mode
    train: Train
    data: Data
    filter: Filter
    dataset_path: Path = Path("./dataset").resolve().absolute()
    weights: Path = Path("./weights").resolve().absolute()
    results: Path = Path("./results").resolve().absolute()
    map_filename: str = "map_{dataset}_{building}.npz"
    network_filename: str = "trajnet_{dataset}_{building}.ptl"
