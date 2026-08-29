"""Cell 9R v2: numerically safe publication-scale ClosureGraph-InSAR repair.

This script trains the paper-aligned UNet-3D-3-S-M baseline and the promoted
Graph + amplitude model on the production train split created by Cell 8.  It
uses the validation split for checkpointing, never opens the locked test
split, repeats one corrected protocol for three independent seeds, rejects
non-finite values instead of converting them to zero, checkpoints every
completed finite epoch, and writes every repaired artifact to a new versioned
location on Google Drive. Existing Cell-9 outputs are never modified.

Upload this file to /content and run it in one Colab cell with:
    %run -i /content/closuregraph_insar_cell9_repair_v2.py

Cell 9 is self-contained apart from the Cell-8 production dataset on Drive.
It does not require Cells 2-8 to remain active in memory.
"""

from __future__ import annotations

from bisect import bisect_right
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import math
import os
import random
import shutil
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from IPython.display import display
except ImportError:
    display = None

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


# =============================================================================
# SMALL DISPLAY AND FILE UTILITIES
# =============================================================================
WIDTH = 104


def banner(text: str) -> None:
    print("=" * WIDTH)
    print(text)
    print("=" * WIDTH)


def stage(number: int, title: str, explanation: str) -> None:
    print("\n" + "-" * WIDTH)
    print(f"STAGE {number}/7 - {title}")
    print(explanation)


def show_table(frame: pd.DataFrame, title: str, digits: int = 6) -> None:
    print(f"\n{title}\n")
    if display is not None:
        try:
            display(frame.round(digits).style.hide(axis="index"))
            return
        except Exception:
            pass
    with pd.option_context(
        "display.max_rows", 100,
        "display.max_columns", None,
        "display.width", 220,
    ):
        print(frame.round(digits).to_string(index=False))


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_figure_atomic(fig, path: Path, dpi: int = 400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(temporary, dpi=dpi, bbox_inches="tight", facecolor="white")
    with Image.open(temporary) as image:
        image.verify()
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# STAGE 1: SESSION AND OUTPUTS
# =============================================================================
banner("CELL 9R v2: NUMERICALLY SAFE MULTI-SEED REPAIR TRAINING")
stage(
    1,
    "Checking the A100 session and Drive dataset.",
    "This stage finds the verified Cell-8 production shards and prepares new\n"
    "repair-specific model, table, figure, and log paths. The original Cell-9\n"
    "checkpoints and results are preserved unchanged.",
)

DRY_RUN = os.environ.get("CLOSUREGRAPH_CELL9R_DRY_RUN", "0") == "1"
FORCE_RETRAIN = os.environ.get("CLOSUREGRAPH_CELL9R_FORCE_RETRAIN", "0") == "1"
SKIP_LOCAL_CACHE = (
    os.environ.get("CLOSUREGRAPH_CELL9R_SKIP_LOCAL_CACHE", "0") == "1"
)

DRIVE_ROOT = Path("/content/drive/MyDrive/ClosureGraph_InSAR")
FALLBACK_ROOT = Path.cwd() / "ClosureGraph_InSAR_test"
if (DRIVE_ROOT / "configs" / "production_dataset_config_v1.json").exists():
    PROJECT_ROOT = DRIVE_ROOT
elif (FALLBACK_ROOT / "configs" / "production_dataset_config_v1.json").exists():
    PROJECT_ROOT = FALLBACK_ROOT
else:
    raise FileNotFoundError(
        "The Cell-8 production dataset was not found. Mount Google Drive and "
        "confirm /content/drive/MyDrive/ClosureGraph_InSAR exists."
    )

DATASET_VERSION = "production_v1"
RUN_VERSION = "production_v1_repair_v2"
DATA_DIR = PROJECT_ROOT / "data" / DATASET_VERSION
CONFIG_SOURCE = PROJECT_ROOT / "configs" / "production_dataset_config_v1.json"
MODEL_DIR = PROJECT_ROOT / "models" / RUN_VERSION
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
LOCAL_CACHE_DIR = Path("/content/closuregraph_training_cache") / DATASET_VERSION

for directory in [MODEL_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

with open(CONFIG_SOURCE, "r", encoding="utf-8") as stream:
    DATA_CONFIG = json.load(stream)

CONFIG_SIGNATURE = str(DATA_CONFIG["configuration_sha256"])
GENERATOR_CONFIG = DATA_CONFIG["generator_config"]
DEPTH = int(GENERATOR_CONFIG["sequence_length"])
HEIGHT = int(GENERATOR_CONFIG["image_height"])
IMAGE_WIDTH = int(GENERATOR_CONFIG["image_width"])
PACKED_WIDTH = int(math.ceil(IMAGE_WIDTH / 8.0))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda" and not DRY_RUN:
    raise RuntimeError(
        "Cell 9 requires an A100 GPU. In Colab select Runtime > Change runtime "
        "type > A100 GPU, then rerun this file."
    )

print(f"Project root       : {PROJECT_ROOT}")
print(f"Dataset version    : {DATASET_VERSION}")
print(f"Repair run version : {RUN_VERSION}")
print(f"Input/output shape : 2x{DEPTH}x{HEIGHT}x{IMAGE_WIDTH} -> 1x{DEPTH}x{HEIGHT}x{IMAGE_WIDTH}")
print(f"Device             : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU                : {torch.cuda.get_device_name(0)}")
print(f"Force retrain      : {FORCE_RETRAIN}")


# =============================================================================
# STAGE 2: FROZEN TRAINING PROTOCOL
# =============================================================================
stage(
    2,
    "Freezing the training and validation protocol.",
    "Three independent seeds quantify training variability. For every seed,\n"
    "the baseline is trained first. A node-only graph warm-up then reproduces\n"
    "the Cell-6 initialization used by the promoted Graph + amplitude model.\n"
    "The final baseline and promoted checkpoints use the same common score.",
)

TRAINING_SEEDS = [31415, 27182, 16180]
BATCH_SIZE = 2
NUMBER_OF_WORKERS = 2
BASE_CHANNELS = 16
TEMPORAL_KERNEL = 3
DROPOUT_PROBABILITY = 0.20
GRAPH_LAGS = (1, 2, 4, 8)
TOP_FRACTION = 0.005
PEAK_WEIGHT = 0.35
TEMPORAL_WEIGHT = 0.10
GRADIENT_CLIP = 1.0

PHASES = {
    "baseline": {
        "display_name": "Baseline",
        "max_epochs": 30,
        "minimum_epochs": 12,
        "patience": 8,
        "learning_rate": 5.0e-4,
        "training_objective": "node_mae",
        "checkpoint_metric": "common_selection_score",
        "initialization": "random",
    },
    "graph_warmup": {
        "display_name": "Graph warm-up (internal)",
        "max_epochs": 25,
        "minimum_epochs": 10,
        "patience": 7,
        "learning_rate": 2.0e-4,
        "training_objective": "node_mae",
        "checkpoint_metric": "node_mae",
        "initialization": "best baseline backbone from the same seed",
    },
    "graph_amplitude": {
        "display_name": "Graph + amplitude",
        "max_epochs": 20,
        "minimum_epochs": 8,
        "patience": 6,
        "learning_rate": 2.0e-4,
        "training_objective": "common_selection_score",
        "checkpoint_metric": "common_selection_score",
        "initialization": "best same-seed node-only graph warm-up",
    },
}

if DRY_RUN:
    TRAINING_SEEDS = [TRAINING_SEEDS[0]]
    for phase_config in PHASES.values():
        phase_config["max_epochs"] = 1
        phase_config["minimum_epochs"] = 1
        phase_config["patience"] = 1

PROTOCOL_PATH = PROJECT_ROOT / "configs" / "closuregraph_cell9_repair_protocol_v2.json"
HISTORY_PATH = TABLE_DIR / "closuregraph_cell9_repair_history_v2.csv"
SUMMARY_PATH = TABLE_DIR / "closuregraph_cell9_repair_validation_summary_v2.csv"
AGGREGATE_PATH = TABLE_DIR / "closuregraph_cell9_repair_validation_aggregate_v2.csv"
TRAINING_FIGURE_PATH = FIGURE_DIR / "closuregraph_cell9_repair_multiseed_training_v2.png"
COMPONENT_FIGURE_PATH = FIGURE_DIR / "closuregraph_cell9_repair_validation_components_v2.png"

protocol_core = {
    "cell": "9R",
    "repair_version": 2,
    "run_version": RUN_VERSION,
    "dataset_version": DATASET_VERSION,
    "dataset_configuration_sha256": CONFIG_SIGNATURE,
    "training_seeds": TRAINING_SEEDS,
    "batch_size": BATCH_SIZE,
    "number_of_workers": NUMBER_OF_WORKERS,
    "base_channels": BASE_CHANNELS,
    "temporal_kernel": TEMPORAL_KERNEL,
    "dropout_probability": DROPOUT_PROBABILITY,
    "graph_lags": GRAPH_LAGS,
    "top_fraction": TOP_FRACTION,
    "peak_weight": PEAK_WEIGHT,
    "temporal_weight": TEMPORAL_WEIGHT,
    "gradient_clip": GRADIENT_CLIP,
    "optimizer": "Adam",
    "mixed_precision": "bfloat16 on supported CUDA GPUs; float32 otherwise",
    "loss_precision": "float32",
    "nonfinite_policy": "raise immediately; never aggregate or checkpoint",
    "augmentation": "deterministic conditional horizontal and vertical flips",
    "checkpoint_selection": (
        "baseline and graph_amplitude: validation common_selection_score; "
        "internal graph_warmup: validation node_mae"
    ),
    "phases": PHASES,
    "locked_test_policy": (
        "Cell 9R never opens locked_test shards. Locked-test evaluation is "
        "deferred until every repaired seed has finite non-zero validation metrics."
    ),
    "dry_run": DRY_RUN,
}
PROTOCOL_SIGNATURE = hashlib.sha256(
    json.dumps(protocol_core, sort_keys=True).encode("utf-8")
).hexdigest()
protocol_payload = {
    **protocol_core,
    "protocol_sha256": PROTOCOL_SIGNATURE,
    "training_complete": False,
}

if PROTOCOL_PATH.exists() and not FORCE_RETRAIN:
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as stream:
        prior_protocol = json.load(stream)
    if prior_protocol.get("protocol_sha256") != PROTOCOL_SIGNATURE:
        raise RuntimeError(
            "The existing Cell-9R v2 protocol differs from this script. Do not "
            "silently mix training protocols. Use a new repair version or set "
            "CLOSUREGRAPH_CELL9R_FORCE_RETRAIN=1 after reviewing the change."
        )
else:
    atomic_json(protocol_payload, PROTOCOL_PATH)

protocol_table = pd.DataFrame(
    [
        {
            "phase": name,
            "model": config["display_name"],
            "seeds": len(TRAINING_SEEDS),
            "max_epochs": config["max_epochs"],
            "minimum_epochs": config["minimum_epochs"],
            "patience": config["patience"],
            "learning_rate": config["learning_rate"],
            "training_objective": config["training_objective"],
            "checkpoint_selection": config["checkpoint_metric"],
        }
        for name, config in PHASES.items()
    ]
)
show_table(protocol_table, "Frozen Cell-9R v2 repair protocol")
print("Locked test         : NOT indexed or opened in Cell 9R")


# =============================================================================
# STAGE 3: TRAIN/VALIDATION SHARD AUDIT AND LOCAL CACHE
# =============================================================================
stage(
    3,
    "Auditing and caching train/validation shards only.",
    "The script verifies every training and validation shard before use. It\n"
    "copies only those two splits to Colab's local disk for faster epochs;\n"
    "locked-test filenames are deliberately excluded from discovery and I/O.",
)


def discover_split_shards(split: str) -> list[Path]:
    if split not in {"train", "validation"}:
        raise ValueError("Cell 9 permits only train and validation splits.")
    paths = sorted(DATA_DIR.glob(f"{split}_{DATASET_VERSION}_*.h5"))
    if not paths:
        raise FileNotFoundError(f"No {split} shards found in {DATA_DIR}")
    if any("locked_test" in str(path) for path in paths):
        raise RuntimeError("Locked-test leakage detected in shard discovery.")
    return paths


def validate_training_shard(path: Path, expected_split: str) -> int:
    if expected_split not in {"train", "validation"}:
        raise ValueError("Locked-test shard access is forbidden in Cell 9.")
    if "locked_test" in path.name:
        raise RuntimeError(f"Cell 9 refused to open locked test path: {path}")
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("status", "") != "complete":
            raise ValueError(f"Incomplete shard: {path}")
        if handle.attrs.get("split", "") != expected_split:
            raise ValueError(f"Incorrect split attribute: {path}")
        if bool(handle.attrs.get("locked_for_final_only", False)):
            raise ValueError(f"Final-only shard refused: {path}")
        if handle.attrs.get("configuration_sha256", "") != CONFIG_SIGNATURE:
            raise ValueError(f"Configuration mismatch: {path}")
        count = int(handle.attrs["number_of_samples"])
        expected_shapes = {
            "observed_normalized": (count, DEPTH, HEIGHT, IMAGE_WIDTH),
            "valid_mask_packed": (count, DEPTH, HEIGHT, PACKED_WIDTH),
            "target_spatial_mm": (count, HEIGHT, IMAGE_WIDTH),
            "target_temporal_profile": (count, DEPTH),
        }
        for key, shape in expected_shapes.items():
            if tuple(handle[key].shape) != shape:
                raise ValueError(f"Unexpected {key} shape in {path}: {handle[key].shape}")
        if handle["metadata/sample_seed"].shape != (count,):
            raise ValueError(f"Invalid seed vector in {path}")
    return count


source_shards = {
    split: discover_split_shards(split)
    for split in ["train", "validation"]
}
source_counts = {
    split: sum(validate_training_shard(path, split) for path in paths)
    for split, paths in source_shards.items()
}

expected_counts = {
    split: int(DATA_CONFIG["split_specifications"][split]["count"])
    for split in ["train", "validation"]
}
if source_counts != expected_counts:
    raise RuntimeError(
        f"Shard counts do not match the Cell-8 configuration: "
        f"found={source_counts}, expected={expected_counts}"
    )


def cache_split(paths: list[Path], split: str) -> list[Path]:
    if SKIP_LOCAL_CACHE or not str(PROJECT_ROOT).startswith("/content/drive/"):
        return paths
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_paths = []
    for source in tqdm(paths, desc=f"Caching {split}", unit="shard"):
        destination = LOCAL_CACHE_DIR / source.name
        copy_required = (
            not destination.exists()
            or destination.stat().st_size != source.stat().st_size
        )
        if copy_required:
            temporary = destination.with_suffix(destination.suffix + ".copying")
            if temporary.exists():
                temporary.unlink()
            shutil.copy2(source, temporary)
            validate_training_shard(temporary, split)
            os.replace(temporary, destination)
        validate_training_shard(destination, split)
        cached_paths.append(destination)
    return cached_paths


TRAIN_SHARDS = cache_split(source_shards["train"], "train")
VALIDATION_SHARDS = cache_split(source_shards["validation"], "validation")

shard_audit = pd.DataFrame(
    [
        {
            "split": split,
            "sequences": source_counts[split],
            "shards": len(source_shards[split]),
            "source": "Google Drive" if str(PROJECT_ROOT).startswith("/content/drive/") else "local test data",
            "training_read_path": (
                "local Colab cache"
                if paths and str(paths[0]).startswith("/content/closuregraph_training_cache")
                else "source directory"
            ),
            "locked_for_final_only": False,
        }
        for split, paths in {
            "train": TRAIN_SHARDS,
            "validation": VALIDATION_SHARDS,
        }.items()
    ]
)
show_table(shard_audit, "Cell-9 data access audit", digits=0)
print("Locked-test shard handles created: 0")


# =============================================================================
# STAGE 4: COMPACT LOADER, MODELS, AND OBJECTIVES
# =============================================================================
stage(
    4,
    "Building the compact loader, models, and frozen objectives.",
    "Each sample is reconstructed as float32 from Cell 8's float16/bit-packed\n"
    "representation. Baseline and graph warm-up minimize MAE. The final graph\n"
    "phase adds the pilot-selected peak-region and temporal-gradient terms.",
)


class ProductionShardDataset(Dataset):
    """Worker-safe reader for compact train/validation shards."""

    def __init__(
        self,
        paths: list[Path],
        split: str,
        augment: bool,
        augmentation_seed: int,
        epoch: int,
        maximum_samples: int | None = None,
    ):
        if split not in {"train", "validation"}:
            raise ValueError("Cell 9 dataset allows train/validation only.")
        self.paths = [str(path) for path in paths]
        if any("locked_test" in path for path in self.paths):
            raise RuntimeError("Locked-test path refused by dataset.")
        self.split = split
        self.augment = bool(augment)
        self.augmentation_seed = int(augmentation_seed)
        self.epoch = int(epoch)
        self._handles: dict[int, h5py.File] = {}
        counts = [validate_training_shard(Path(path), split) for path in self.paths]
        self.cumulative = np.cumsum(counts).tolist()
        full_length = int(self.cumulative[-1])
        self.length = min(full_length, maximum_samples or full_length)

    def __len__(self) -> int:
        return self.length

    def _handle(self, shard_index: int) -> h5py.File:
        if shard_index not in self._handles:
            self._handles[shard_index] = h5py.File(
                self.paths[shard_index], "r", swmr=True
            )
        return self._handles[shard_index]

    def __getitem__(self, global_index: int):
        shard_index = bisect_right(self.cumulative, int(global_index))
        previous = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        local_index = int(global_index) - int(previous)
        handle = self._handle(shard_index)

        observed = handle["observed_normalized"][local_index].astype(np.float32)
        packed_mask = handle["valid_mask_packed"][local_index]
        valid_mask = np.unpackbits(
            packed_mask,
            axis=-1,
            count=IMAGE_WIDTH,
            bitorder="little",
        ).astype(np.float32)
        spatial_mm = handle["target_spatial_mm"][local_index].astype(np.float32)
        temporal = handle["target_temporal_profile"][local_index].astype(np.float32)
        target_mm = temporal[:, None, None] * spatial_mm[None, :, :]

        scale_min = np.float32(handle["metadata/scale_min_mm"][local_index])
        scale_max = np.float32(handle["metadata/scale_max_mm"][local_index])
        scale_range = max(float(scale_max - scale_min), 1.0e-6)
        target = (2.0 * (target_mm - scale_min) / scale_range - 1.0)[None]
        model_input = np.stack([observed, valid_mask], axis=0)

        if self.augment:
            mixed_seed = (
                self.augmentation_seed * 1_000_003
                + self.epoch * 97_409
                + int(global_index) * 65_537
            ) % (2**32)
            generator = np.random.default_rng(mixed_seed)
            if generator.random() >= 0.5:
                model_input = np.flip(model_input, axis=-1)
                target = np.flip(target, axis=-1)
            if generator.random() >= 0.5:
                model_input = np.flip(model_input, axis=-2)
                target = np.flip(target, axis=-2)

        sample_seed = np.int64(handle["metadata/sample_seed"][local_index])
        return {
            "input": torch.from_numpy(np.ascontiguousarray(model_input)),
            "target": torch.from_numpy(np.ascontiguousarray(target.astype(np.float32))),
            "scale_min_mm": torch.tensor(scale_min),
            "scale_max_mm": torch.tensor(scale_max),
            "sample_seed": torch.tensor(sample_seed),
            "sample_index": torch.tensor(global_index, dtype=torch.long),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self):
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass


class SeparableConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.spatial_depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=in_channels,
            bias=False,
        )
        self.temporal_pointwise = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(TEMPORAL_KERNEL, 1, 1),
            padding=(TEMPORAL_KERNEL // 2, 0, 0),
            bias=False,
        )
        self.activation = nn.PReLU(num_parameters=out_channels)
        self.normalization = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=DROPOUT_PROBABILITY)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.spatial_depthwise(tensor)
        tensor = self.temporal_pointwise(tensor)
        tensor = self.activation(tensor)
        tensor = self.normalization(tensor)
        return self.dropout(tensor)


class DoubleSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            SeparableConv3d(in_channels, out_channels),
            SeparableConv3d(out_channels, out_channels),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


class UNet3D3SM(nn.Module):
    """Paper-aligned 3D encoder-decoder with spatial-only pooling."""

    def __init__(self, base_channels: int = BASE_CHANNELS):
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]
        self.channels = channels
        self.pool = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.encoder_1 = DoubleSeparableBlock(2, channels[0])
        self.encoder_2 = DoubleSeparableBlock(channels[0], channels[1])
        self.encoder_3 = DoubleSeparableBlock(channels[1], channels[2])
        self.encoder_4 = DoubleSeparableBlock(channels[2], channels[3])
        self.bottleneck = DoubleSeparableBlock(channels[3], channels[4])
        self.up_4 = nn.ConvTranspose3d(channels[4], channels[3], (1, 2, 2), (1, 2, 2))
        self.decoder_4 = DoubleSeparableBlock(channels[3] * 2, channels[3])
        self.up_3 = nn.ConvTranspose3d(channels[3], channels[2], (1, 2, 2), (1, 2, 2))
        self.decoder_3 = DoubleSeparableBlock(channels[2] * 2, channels[2])
        self.up_2 = nn.ConvTranspose3d(channels[2], channels[1], (1, 2, 2), (1, 2, 2))
        self.decoder_2 = DoubleSeparableBlock(channels[1] * 2, channels[1])
        self.up_1 = nn.ConvTranspose3d(channels[1], channels[0], (1, 2, 2), (1, 2, 2))
        self.decoder_1 = DoubleSeparableBlock(channels[0] * 2, channels[0])
        self.output_layer = nn.Conv3d(channels[0], 1, kernel_size=1)

    def decode_features(self, tensor: torch.Tensor) -> torch.Tensor:
        encoder_1 = self.encoder_1(tensor)
        encoder_2 = self.encoder_2(self.pool(encoder_1))
        encoder_3 = self.encoder_3(self.pool(encoder_2))
        encoder_4 = self.encoder_4(self.pool(encoder_3))
        bottleneck = self.bottleneck(self.pool(encoder_4))
        decoder_4 = self.decoder_4(torch.cat([self.up_4(bottleneck), encoder_4], dim=1))
        decoder_3 = self.decoder_3(torch.cat([self.up_3(decoder_4), encoder_3], dim=1))
        decoder_2 = self.decoder_2(torch.cat([self.up_2(decoder_3), encoder_2], dim=1))
        return self.decoder_1(torch.cat([self.up_1(decoder_2), encoder_1], dim=1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.decode_features(tensor))


class ReliabilityWeightedTemporalGraph(nn.Module):
    def __init__(self, channels: int, lags: tuple[int, ...]):
        super().__init__()
        self.lags = tuple(int(lag) for lag in lags)
        self.message_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size=1, bias=False),
                    nn.PReLU(num_parameters=channels),
                    nn.BatchNorm3d(channels),
                )
                for _ in self.lags
            ]
        )
        self.lag_strengths = nn.Parameter(torch.zeros(len(self.lags)))
        self.update = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.PReLU(num_parameters=channels),
            nn.BatchNorm3d(channels),
            nn.Dropout3d(p=0.10),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        aggregate = torch.zeros_like(features)
        normalizer = torch.zeros_like(valid_mask)
        strengths = F.softplus(self.lag_strengths) + 1.0e-4
        for lag_index, lag in enumerate(self.lags):
            if lag >= features.shape[2]:
                continue
            projected = self.message_projections[lag_index](features)
            strength = strengths[lag_index] / float(lag)
            earlier_validity = valid_mask[:, :, :-lag]
            later_validity = valid_mask[:, :, lag:]
            aggregate = aggregate + F.pad(
                strength * projected[:, :, :-lag] * earlier_validity,
                (0, 0, 0, 0, lag, 0),
            )
            normalizer = normalizer + F.pad(
                strength * earlier_validity,
                (0, 0, 0, 0, lag, 0),
            )
            aggregate = aggregate + F.pad(
                strength * projected[:, :, lag:] * later_validity,
                (0, 0, 0, 0, 0, lag),
            )
            normalizer = normalizer + F.pad(
                strength * later_validity,
                (0, 0, 0, 0, 0, lag),
            )
        aggregate = aggregate / normalizer.clamp_min(1.0e-4)
        combined = torch.cat([features, aggregate - features], dim=1)
        proposal = self.update(combined)
        gate = self.gate(combined)
        return features + gate * proposal


class GraphAmplitudeInSAR(UNet3D3SM):
    def __init__(self, base_channels: int = BASE_CHANNELS):
        super().__init__(base_channels=base_channels)
        self.temporal_graph = ReliabilityWeightedTemporalGraph(
            self.channels[0], GRAPH_LAGS
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.decode_features(tensor)
        graph_features = self.temporal_graph(features, tensor[:, 1:2])
        return self.output_layer(graph_features)


def peak_region_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch, channels, epochs, height, width = target.shape
    pixels = height * width
    selected = max(1, int(round(TOP_FRACTION * pixels)))
    prediction_flat = prediction.reshape(batch, channels, epochs, pixels).reshape(-1, pixels)
    target_flat = target.reshape(batch, channels, epochs, pixels).reshape(-1, pixels)
    indices = torch.topk(
        target_flat.abs(), selected, dim=1, largest=True, sorted=False
    ).indices
    return F.l1_loss(
        torch.gather(prediction_flat, 1, indices),
        torch.gather(target_flat, 1, indices),
    )


def objective_components(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    node_mae = F.l1_loss(prediction, target)
    peak_loss = peak_region_loss(prediction, target)
    temporal_loss = F.l1_loss(
        prediction[:, :, 1:] - prediction[:, :, :-1],
        target[:, :, 1:] - target[:, :, :-1],
    )
    common_score = node_mae + PEAK_WEIGHT * peak_loss + TEMPORAL_WEIGHT * temporal_loss
    return {
        "node_mae": node_mae,
        "peak_loss": peak_loss,
        "temporal_loss": temporal_loss,
        "common_selection_score": common_score,
    }


def transfer_baseline_to_graph(graph_model: nn.Module, baseline_state: dict) -> None:
    result = graph_model.load_state_dict(baseline_state, strict=False)
    invalid_missing = [
        key for key in result.missing_keys if not key.startswith("temporal_graph.")
    ]
    if invalid_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Invalid baseline-to-graph transfer: missing={invalid_missing}, "
            f"unexpected={result.unexpected_keys}"
        )


baseline_parameter_count = sum(
    parameter.numel() for parameter in UNet3D3SM().parameters()
)
graph_parameter_count = sum(
    parameter.numel() for parameter in GraphAmplitudeInSAR().parameters()
)

model_table = pd.DataFrame(
    [
        {
            "model": PHASES["baseline"]["display_name"],
            "parameters": baseline_parameter_count,
            "training_loss": "node MAE",
            "checkpoint_metric": "common validation score",
        },
        {
            "model": PHASES["graph_warmup"]["display_name"],
            "parameters": graph_parameter_count,
            "training_loss": "node MAE",
            "checkpoint_metric": "node MAE (initialization only)",
        },
        {
            "model": PHASES["graph_amplitude"]["display_name"],
            "parameters": graph_parameter_count,
            "training_loss": "node + 0.35 peak + 0.10 temporal",
            "checkpoint_metric": "common validation score",
        },
    ]
)
show_table(model_table, "Production models and frozen objectives", digits=0)

sample_check = ProductionShardDataset(
    TRAIN_SHARDS,
    "train",
    augment=False,
    augmentation_seed=TRAINING_SEEDS[0],
    epoch=0,
    maximum_samples=1,
)[0]
if tuple(sample_check["input"].shape) != (2, DEPTH, HEIGHT, IMAGE_WIDTH):
    raise RuntimeError("Compact input reconstruction failed.")
if tuple(sample_check["target"].shape) != (1, DEPTH, HEIGHT, IMAGE_WIDTH):
    raise RuntimeError("Compact target reconstruction failed.")
if not torch.isfinite(sample_check["input"]).all() or not torch.isfinite(sample_check["target"]).all():
    raise RuntimeError("Non-finite compact sample reconstructed.")
print("Compact sample check: PASSED (float32 input/target, binary valid mask)")

if DRY_RUN:
    banner("CELL 9R v2 DRY RUN COMPLETE: PREFLIGHT AND MODEL DEFINITIONS PASSED")
    raise SystemExit(0)


# =============================================================================
# STAGE 5: RESTART-SAFE MULTI-SEED TRAINING
# =============================================================================
stage(
    5,
    "Training the baseline, graph warm-up, and promoted model for every seed.",
    "One compact line is printed per epoch. The last state and best validation\n"
    "state are written atomically to Drive, so rerunning this same file resumes\n"
    "after a disconnect without repeating completed epochs.",
)

BF16_ENABLED = (
    DEVICE.type == "cuda"
    and hasattr(torch.cuda, "is_bf16_supported")
    and torch.cuda.is_bf16_supported()
)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

print(
    "Training precision  : "
    + ("bfloat16 autocast with float32 losses" if BF16_ENABLED else "float32")
)
print("Non-finite policy   : immediate error; invalid values are never averaged")


def autocast_context():
    if BF16_ENABLED:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def assert_finite_tensor(
    name: str,
    tensor: torch.Tensor,
    phase_name: str,
    run_seed: int,
    epoch: int,
    split: str,
) -> None:
    if not torch.isfinite(tensor).all():
        finite = torch.isfinite(tensor)
        finite_count = int(finite.sum().detach().cpu())
        total_count = int(tensor.numel())
        raise FloatingPointError(
            f"Non-finite {name} detected in {split}: phase={phase_name}, "
            f"seed={run_seed}, epoch={epoch}, finite={finite_count}/{total_count}. "
            "No checkpoint was written for this epoch. Rerunning this file "
            "will resume from the preceding finite epoch."
        )


def assert_model_finite(
    model: nn.Module,
    phase_name: str,
    run_seed: int,
    epoch: int,
    split: str,
) -> None:
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(
                f"Non-finite model state '{name}' detected after {split}: "
                f"phase={phase_name}, seed={run_seed}, epoch={epoch}. "
                "No checkpoint was written for this epoch."
            )


def validate_checkpoint_payload(checkpoint: dict, path: Path) -> None:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint has no model state: {path}")
    for name, value in state.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise RuntimeError(
                f"Checkpoint contains a non-finite model tensor '{name}': {path}"
            )
    components = checkpoint.get("validation_components")
    if components is not None:
        required = {
            "node_mae",
            "peak_loss",
            "temporal_loss",
            "common_selection_score",
        }
        missing = required.difference(components)
        if missing:
            raise RuntimeError(
                f"Checkpoint validation components are incomplete ({missing}): {path}"
            )
        for name in sorted(required):
            value = float(components[name])
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(
                    f"Checkpoint validation metric '{name}' is invalid "
                    f"({value!r}): {path}"
                )


def make_loader(split: str, run_seed: int, epoch: int) -> DataLoader:
    training = split == "train"
    paths = TRAIN_SHARDS if training else VALIDATION_SHARDS
    maximum_samples = None
    dataset = ProductionShardDataset(
        paths,
        split,
        augment=training,
        augmentation_seed=run_seed,
        epoch=epoch,
        maximum_samples=maximum_samples,
    )
    generator = torch.Generator()
    generator.manual_seed(run_seed + epoch * 104_729)
    arguments = {
        "batch_size": BATCH_SIZE,
        "shuffle": training,
        "drop_last": False,
        "num_workers": NUMBER_OF_WORKERS,
        "pin_memory": True,
        "persistent_workers": False,
        "generator": generator,
    }
    if NUMBER_OF_WORKERS > 0:
        arguments["prefetch_factor"] = 2
    return DataLoader(dataset, **arguments)


def run_epoch(
    model: nn.Module,
    phase_name: str,
    run_seed: int,
    epoch: int,
    optimizer,
    scaler,
    training: bool,
) -> dict[str, float]:
    split = "train" if training else "validation"
    loader = make_loader(split, run_seed, epoch)
    model.train(training)
    sums = {
        "node_mae": 0.0,
        "peak_loss": 0.0,
        "temporal_loss": 0.0,
        "common_selection_score": 0.0,
    }
    metric_sample_counts = {key: 0 for key in sums}
    processed_samples = 0

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in loader:
            inputs = batch["input"].to(DEVICE, non_blocking=True)
            targets = batch["target"].to(DEVICE, non_blocking=True)
            batch_samples = int(inputs.shape[0])
            assert_finite_tensor(
                "input", inputs, phase_name, run_seed, epoch, split
            )
            assert_finite_tensor(
                "target", targets, phase_name, run_seed, epoch, split
            )
            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                prediction = model(inputs)
            prediction = prediction.float()
            targets_float = targets.float()
            assert_finite_tensor(
                "prediction", prediction, phase_name, run_seed, epoch, split
            )

            # All objectives and reported metrics are computed in float32.
            need_all = (phase_name == "graph_amplitude") or (not training)
            if need_all:
                components = objective_components(prediction, targets_float)
                active_metric_keys = tuple(sums)
            else:
                node_mae = F.l1_loss(prediction, targets_float)
                components = {
                    "node_mae": node_mae,
                    "common_selection_score": node_mae,
                }
                active_metric_keys = ("node_mae", "common_selection_score")
            loss = (
                components["common_selection_score"]
                if phase_name == "graph_amplitude"
                else components["node_mae"]
            )

            for key in active_metric_keys:
                assert_finite_tensor(
                    key,
                    components[key],
                    phase_name,
                    run_seed,
                    epoch,
                    split,
                )

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRADIENT_CLIP,
                    error_if_nonfinite=True,
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm in training: phase={phase_name}, "
                        f"seed={run_seed}, epoch={epoch}."
                    )
                scaler.step(optimizer)
                scaler.update()

            for key in active_metric_keys:
                value = float(components[key].detach().cpu())
                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"Non-finite scalar metric '{key}' in {split}: "
                        f"phase={phase_name}, seed={run_seed}, epoch={epoch}."
                    )
                sums[key] += value * batch_samples
                metric_sample_counts[key] += batch_samples
            processed_samples += batch_samples

    if processed_samples != len(loader.dataset):
        raise RuntimeError(
            f"Incomplete {split} epoch: processed={processed_samples}, "
            f"expected={len(loader.dataset)}, phase={phase_name}, "
            f"seed={run_seed}, epoch={epoch}."
        )
    required_metric_keys = tuple(sums) if not training else ("node_mae",)
    for key in required_metric_keys:
        if metric_sample_counts[key] != processed_samples:
            raise RuntimeError(
                f"Metric '{key}' did not cover the complete {split} epoch: "
                f"counted={metric_sample_counts[key]}, processed={processed_samples}."
            )

    assert_model_finite(model, phase_name, run_seed, epoch, split)

    return {
        key: (
            value / metric_sample_counts[key]
            if metric_sample_counts[key] > 0
            else float("nan")
        )
        for key, value in sums.items()
    }


def checkpoint_paths(phase_name: str, run_seed: int) -> tuple[Path, Path, Path]:
    # The repair version is part of every filename. This guarantees that no
    # original Cell-9 checkpoint or per-phase history can be overwritten.
    stem = f"closuregraph_{phase_name}_{RUN_VERSION}_seed{run_seed}"
    return (
        MODEL_DIR / f"{stem}_best.pt",
        MODEL_DIR / f"{stem}_last.pt",
        TABLE_DIR / f"{stem}_history.csv",
    )


def create_phase_model(phase_name: str, run_seed: int) -> nn.Module:
    seed_everything(run_seed)
    if phase_name == "baseline":
        return UNet3D3SM(BASE_CHANNELS).to(DEVICE)
    model = GraphAmplitudeInSAR(BASE_CHANNELS).to(DEVICE)
    if phase_name == "graph_amplitude":
        warmup_best, _, _ = checkpoint_paths("graph_warmup", run_seed)
        if not warmup_best.exists():
            raise FileNotFoundError(
                f"The same-seed graph warm-up checkpoint is required: {warmup_best}"
            )
        warmup_checkpoint = safe_torch_load(warmup_best, DEVICE)
        validate_checkpoint_payload(warmup_checkpoint, warmup_best)
        model.load_state_dict(warmup_checkpoint["model_state_dict"], strict=True)
        assert_model_finite(model, phase_name, run_seed, 0, "initialization")
        return model
    baseline_best, _, _ = checkpoint_paths("baseline", run_seed)
    if not baseline_best.exists():
        raise FileNotFoundError(
            f"The same-seed baseline checkpoint is required: {baseline_best}"
        )
    baseline_checkpoint = safe_torch_load(baseline_best, DEVICE)
    validate_checkpoint_payload(baseline_checkpoint, baseline_best)
    transfer_baseline_to_graph(model, baseline_checkpoint["model_state_dict"])
    assert_model_finite(model, phase_name, run_seed, 0, "initialization")
    return model


def train_phase(phase_name: str, run_seed: int) -> tuple[pd.DataFrame, dict, Path]:
    phase_config = PHASES[phase_name]
    best_path, last_path, phase_history_path = checkpoint_paths(phase_name, run_seed)

    if FORCE_RETRAIN:
        for old_path in [best_path, last_path, phase_history_path]:
            if old_path.exists():
                old_path.unlink()

    model = create_phase_model(phase_name, run_seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(phase_config["learning_rate"]), weight_decay=0.0
    )
    # BF16 does not need gradient scaling. Keeping a disabled scaler preserves
    # the uniform training path without reintroducing float16 overflow risk.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    signature = {
        "protocol_sha256": PROTOCOL_SIGNATURE,
        "phase": phase_name,
        "seed": run_seed,
        "dataset_configuration_sha256": CONFIG_SIGNATURE,
    }
    start_epoch = 1
    best_validation_score = math.inf
    epochs_without_improvement = 0
    history_records: list[dict] = []

    if last_path.exists() and not FORCE_RETRAIN:
        checkpoint = safe_torch_load(last_path, DEVICE)
        validate_checkpoint_payload(checkpoint, last_path)
        if checkpoint.get("signature") != signature:
            raise RuntimeError(f"Incompatible existing checkpoint: {last_path}")
        if checkpoint.get("training_complete", False):
            best_checkpoint = safe_torch_load(best_path, DEVICE)
            validate_checkpoint_payload(best_checkpoint, best_path)
            if best_checkpoint.get("signature") != signature:
                raise RuntimeError(f"Incompatible existing checkpoint: {best_path}")
            if phase_history_path.exists():
                history_frame = pd.read_csv(phase_history_path)
            else:
                history_frame = pd.DataFrame(checkpoint.get("history_records", []))
                if history_frame.empty:
                    raise RuntimeError(
                        f"Completed checkpoint has no recoverable history: {last_path}"
                    )
                atomic_csv(history_frame, phase_history_path)
            print(
                f"Using completed {phase_config['display_name']} seed {run_seed}; "
                f"best epoch {best_checkpoint['epoch']}."
            )
            return history_frame, best_checkpoint, best_path
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_score = float(checkpoint["best_validation_score"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if checkpoint.get("history_records"):
            history_records = list(checkpoint["history_records"])
        elif phase_history_path.exists():
            history_records = pd.read_csv(phase_history_path).to_dict("records")
        print(
            f"Resuming {phase_config['display_name']} seed {run_seed} "
            f"after epoch {start_epoch - 1}."
        )

    print("\n" + "-" * 92)
    print(f"TRAINING: {phase_config['display_name']} | seed {run_seed}")
    print("-" * 92)

    for epoch in range(start_epoch, int(phase_config["max_epochs"]) + 1):
        epoch_start = time.time()
        train_values = run_epoch(
            model, phase_name, run_seed, epoch, optimizer, scaler, training=True
        )
        validation_values = run_epoch(
            model, phase_name, run_seed, epoch, optimizer, scaler, training=False
        )
        checkpoint_metric = str(phase_config["checkpoint_metric"])
        validation_score = validation_values[checkpoint_metric]
        if not math.isfinite(validation_score) or validation_score <= 0.0:
            raise FloatingPointError(
                f"Invalid validation checkpoint metric ({validation_score!r}): "
                f"phase={phase_name}, seed={run_seed}, epoch={epoch}. "
                "No checkpoint was written for this epoch."
            )
        improved = validation_score < best_validation_score - 1.0e-6
        if improved:
            best_validation_score = validation_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record = {
            "phase": phase_name,
            "display_name": phase_config["display_name"],
            "seed": run_seed,
            "epoch": epoch,
            "train_node_mae": train_values["node_mae"],
            "validation_node_mae": validation_values["node_mae"],
            "validation_peak_loss": validation_values["peak_loss"],
            "validation_temporal_loss": validation_values["temporal_loss"],
            "validation_common_score": validation_values["common_selection_score"],
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_metric_value": validation_score,
            "best_checkpoint_metric_value": best_validation_score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_start,
            "best_checkpoint": bool(improved),
        }
        history_records.append(record)
        checkpoint = {
            "signature": signature,
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_score": float(best_validation_score),
            "epochs_without_improvement": int(epochs_without_improvement),
            "validation_components": validation_values,
            "history_records": history_records,
            "training_complete": False,
        }
        if improved:
            atomic_checkpoint(checkpoint, best_path)
        atomic_checkpoint(checkpoint, last_path)

        atomic_csv(pd.DataFrame(history_records), phase_history_path)

        marker = " BEST" if improved else ""
        print(
            f"{epoch:02d}/{phase_config['max_epochs']:02d} | "
            f"node {train_values['node_mae']:.5f}/"
            f"{validation_values['node_mae']:.5f} | "
            f"peak {validation_values['peak_loss']:.5f} | "
            f"select {checkpoint_metric} {validation_score:.5f} | "
            f"{record['epoch_seconds']:.1f} s{marker}"
        )

        if (
            epoch >= int(phase_config["minimum_epochs"])
            and epochs_without_improvement >= int(phase_config["patience"])
        ):
            print(
                f"Early stopping after {phase_config['patience']} epochs "
                "without validation improvement."
            )
            break

    if not best_path.exists():
        raise RuntimeError(f"No best checkpoint was produced: {best_path}")
    final_last = safe_torch_load(last_path, DEVICE)
    validate_checkpoint_payload(final_last, last_path)
    final_last["training_complete"] = True
    atomic_checkpoint(final_last, last_path)
    best_checkpoint = safe_torch_load(best_path, DEVICE)
    validate_checkpoint_payload(best_checkpoint, best_path)
    return pd.DataFrame(history_records), best_checkpoint, best_path


all_histories = []
summary_rows = []
training_start = time.time()

for run_seed in TRAINING_SEEDS:
    for phase_name in ["baseline", "graph_warmup", "graph_amplitude"]:
        history_frame, best_checkpoint, best_path = train_phase(phase_name, run_seed)
        all_histories.append(history_frame)
        components = best_checkpoint["validation_components"]
        summary_rows.append(
            {
                "phase": phase_name,
                "model": PHASES[phase_name]["display_name"],
                "seed": run_seed,
                "parameters": (
                    baseline_parameter_count
                    if phase_name == "baseline"
                    else graph_parameter_count
                ),
                "best_epoch": int(best_checkpoint["epoch"]),
                "validation_node_mae": float(components["node_mae"]),
                "validation_peak_loss": float(components["peak_loss"]),
                "validation_temporal_loss": float(components["temporal_loss"]),
                "validation_common_score": float(components["common_selection_score"]),
                "best_checkpoint": str(best_path),
            }
        )

cell9_history = pd.concat(all_histories, ignore_index=True)
cell9_summary = pd.DataFrame(summary_rows)

metric_columns = [
    "validation_node_mae",
    "validation_peak_loss",
    "validation_temporal_loss",
    "validation_common_score",
]
expected_runs = len(TRAINING_SEEDS) * len(PHASES)
if len(cell9_summary) != expected_runs:
    raise RuntimeError(
        f"Incomplete repair summary: found {len(cell9_summary)} runs, "
        f"expected {expected_runs}."
    )
for metric in metric_columns:
    values = cell9_summary[metric].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise RuntimeError(
            f"Repair summary contains invalid values in '{metric}': {values}. "
            "Cell 10 must not be run."
        )

atomic_csv(cell9_history, HISTORY_PATH)
atomic_csv(cell9_summary, SUMMARY_PATH)


# =============================================================================
# STAGE 6: VALIDATION-ONLY MULTI-SEED SUMMARY
# =============================================================================
stage(
    6,
    "Summarizing validation performance without touching the locked test.",
    "The table includes the auditable graph warm-up used only for initialization.\n"
    "Final baseline and promoted results are summarized across seeds. These are\n"
    "validation diagnostics, not final test claims.",
)

show_table(
    cell9_summary[
        [
            "model",
            "seed",
            "parameters",
            "best_epoch",
            "validation_node_mae",
            "validation_peak_loss",
            "validation_temporal_loss",
            "validation_common_score",
        ]
    ],
    "Best validation checkpoint for each independent run",
)

aggregate_rows = []
for (phase_name, model_name), frame in cell9_summary.groupby(["phase", "model"], sort=False):
    row = {
        "phase": phase_name,
        "model": model_name,
        "seeds": len(frame),
        "mean_best_epoch": frame["best_epoch"].mean(),
    }
    for metric in metric_columns:
        row[f"{metric}_mean"] = frame[metric].mean()
        row[f"{metric}_sd"] = frame[metric].std(ddof=1)
    aggregate_rows.append(row)

cell9_aggregate = pd.DataFrame(aggregate_rows)
atomic_csv(cell9_aggregate, AGGREGATE_PATH)
show_table(cell9_aggregate, "Validation aggregate across training seeds")

protocol_payload["training_complete"] = True
protocol_payload["completed_seeds"] = TRAINING_SEEDS
protocol_payload["summary_path"] = str(SUMMARY_PATH)
protocol_payload["aggregate_path"] = str(AGGREGATE_PATH)
protocol_payload["locked_test_opened"] = False
protocol_payload["elapsed_hours_this_invocation"] = (
    time.time() - training_start
) / 3600.0
atomic_json(protocol_payload, PROTOCOL_PATH)

print("Locked-test prediction batches: 0")
print("Training protocol status       : FROZEN")


# =============================================================================
# STAGE 7: DISPLAY AND SAVE TRAINING FIGURES
# =============================================================================
stage(
    7,
    "Displaying and saving the multi-seed training figures.",
    "The first figure shows convergence for every seed. The second compares\n"
    "the validation components at the saved checkpoints. Both are displayed\n"
    "below and also saved to Drive at 400 dpi.",
)

colours = {
    "baseline": "#777777",
    "graph_warmup": "#2C7FB8",
    "graph_amplitude": "#009E73",
}
fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
plot_specs = [
    ("validation_common_score", "Validation common score"),
    ("validation_node_mae", "Validation node MAE"),
    ("validation_peak_loss", "Validation peak-region loss"),
    ("validation_temporal_loss", "Validation temporal-gradient loss"),
]
for axis, (metric, title) in zip(axes.ravel(), plot_specs):
    for (phase_name, seed), frame in cell9_history.groupby(["phase", "seed"], sort=False):
        axis.plot(
            frame["epoch"],
            frame[metric],
            color=colours[phase_name],
            alpha=0.55,
            linewidth=1.4,
            label=f"{PHASES[phase_name]['display_name']} seed {seed}",
        )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized loss")
    axis.grid(alpha=0.22)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
fig.suptitle(
    "ClosureGraph-InSAR production training: independent seeds",
    fontsize=17,
    fontweight="bold",
)
save_figure_atomic(fig, TRAINING_FIGURE_PATH)
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), constrained_layout=True)
component_specs = [
    ("validation_node_mae", "Node MAE"),
    ("validation_peak_loss", "Peak-region loss"),
    ("validation_temporal_loss", "Temporal-gradient loss"),
    ("validation_common_score", "Common score"),
]
model_order = ["baseline", "graph_amplitude"]
for axis, (metric, title) in zip(axes, component_specs):
    means = [
        cell9_summary.loc[cell9_summary["phase"] == phase, metric].mean()
        for phase in model_order
    ]
    standard_deviations = [
        cell9_summary.loc[cell9_summary["phase"] == phase, metric].std(ddof=1)
        for phase in model_order
    ]
    bars = axis.bar(
        [0, 1],
        means,
        yerr=standard_deviations,
        capsize=5,
        color=[colours[phase] for phase in model_order],
        alpha=0.95,
    )
    axis.set_xticks([0, 1], ["Baseline", "Graph +\namplitude"])
    axis.set_title(title)
    axis.set_ylabel("Normalized validation loss")
    axis.grid(axis="y", alpha=0.22)
    for bar, mean in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

fig.suptitle(
    "Frozen-checkpoint validation components (mean ± SD across seeds)",
    fontsize=16,
    fontweight="bold",
)
save_figure_atomic(fig, COMPONENT_FIGURE_PATH)
plt.show()

banner("CELL 9R v2 COMPLETE: NUMERICALLY SAFE REPAIR FINISHED")
print(f"Training sequences        : {source_counts['train']:,}")
print(f"Validation sequences      : {source_counts['validation']:,}")
print(f"Independent seeds         : {len(TRAINING_SEEDS)}")
print("Locked-test sequences read: 0")
print("Final-test claim made     : NO")
print("-" * WIDTH)
for row in cell9_summary.itertuples(index=False):
    print(
        f"{row.model:18s} seed {row.seed} | best epoch {row.best_epoch:02d} | "
        f"val score {row.validation_common_score:.6f}"
    )
print("-" * WIDTH)
print(f"Protocol             : {PROTOCOL_PATH}")
print(f"Training history      : {HISTORY_PATH}")
print(f"Validation summary    : {SUMMARY_PATH}")
print(f"Seed aggregate        : {AGGREGATE_PATH}")
print(f"Training figure       : {TRAINING_FIGURE_PATH}")
print(f"Component figure      : {COMPONENT_FIGURE_PATH}")
print(f"Checkpoint directory  : {MODEL_DIR}")
print("=" * WIDTH)
print(
    "INTERPRETATION: Every repaired validation component is finite and "
    "strictly positive. Cell 10 may now evaluate these repaired frozen "
    "checkpoints on the locked test exactly once."
)
