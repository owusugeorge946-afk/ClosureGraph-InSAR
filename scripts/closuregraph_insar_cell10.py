"""Cell 10: one-time locked-test evaluation for ClosureGraph-InSAR.

Upload to /content and run in one Colab cell:
    %run -i /content/closuregraph_insar_cell10.py

This file never trains.  It streams the Cell-8 locked_test split only after
the three finite Cell-9R-v2 baseline and Graph+amplitude checkpoints have
been verified.  Per-batch records are atomically written to Drive, so an
interrupted evaluation resumes without re-evaluating saved batches.  Once the
final audit is complete, later runs only print the saved result locations and
do not open the locked test again.
"""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
import hashlib
import json
import math
import os
import random
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

try:
    from IPython.display import display
except ImportError:
    display = None

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


WIDTH = 104
DATASET_VERSION = "production_v1"
RUN_VERSION = "production_v1_repair_v2"
CELL10_VERSION = "locked_test_v1"
TRAINING_SEEDS = (31415, 27182, 16180)
BASE_CHANNELS = 16
TEMPORAL_KERNEL = 3
DROPOUT_PROBABILITY = 0.20
GRAPH_LAGS = (1, 2, 4, 8)
TEST_BATCH_SIZE = 2  # conservative for an A100 and safe for a T4 if needed
BOOTSTRAP_REPLICATES = 4000


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


def save_figure_atomic(fig, path: Path, dpi: int = 400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(temporary, dpi=dpi, bbox_inches="tight", facecolor="white")
    os.replace(temporary, path)


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def signature_of(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


banner("CELL 10: ONE-TIME LOCKED-TEST CLOSUREGRAPH-INSAR EVALUATION")
stage(
    1,
    "Checking the completed Cell-9R checkpoints and locked-test policy.",
    "Cell 10 performs no training and never uses the locked test for model selection. "
    "It evaluates the frozen baseline and Graph + amplitude ensembles exactly once.",
)

DRIVE_ROOT = Path("/content/drive/MyDrive/ClosureGraph_InSAR")
FALLBACK_ROOT = Path.cwd() / "ClosureGraph_InSAR_test"
CONFIG_SOURCE_NAME = "production_dataset_config_v1.json"
if (DRIVE_ROOT / "configs" / CONFIG_SOURCE_NAME).exists():
    PROJECT_ROOT = DRIVE_ROOT
elif (FALLBACK_ROOT / "configs" / CONFIG_SOURCE_NAME).exists():
    PROJECT_ROOT = FALLBACK_ROOT
else:
    raise FileNotFoundError("Cell-8 production data were not found. Mount Drive first.")

DATA_DIR = PROJECT_ROOT / "data" / DATASET_VERSION
MODEL_DIR = PROJECT_ROOT / "models" / RUN_VERSION
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
CONFIG_SOURCE = PROJECT_ROOT / "configs" / CONFIG_SOURCE_NAME
PROTOCOL_SOURCE = PROJECT_ROOT / "configs" / "closuregraph_cell9_repair_protocol_v2.json"
BATCH_DIR = TABLE_DIR / f"closuregraph_cell10_locked_test_batches_{CELL10_VERSION}"
RECORDS_PATH = TABLE_DIR / f"closuregraph_cell10_locked_test_records_{CELL10_VERSION}.csv"
SUMMARY_PATH = TABLE_DIR / f"closuregraph_cell10_locked_test_summary_{CELL10_VERSION}.csv"
SIGNIFICANCE_PATH = TABLE_DIR / f"closuregraph_cell10_locked_test_significance_{CELL10_VERSION}.csv"
SEED_SUMMARY_PATH = TABLE_DIR / f"closuregraph_cell10_locked_test_seed_summary_{CELL10_VERSION}.csv"
CONFIG_PATH = LOG_DIR / f"closuregraph_cell10_locked_test_config_{CELL10_VERSION}.json"
LEDGER_PATH = LOG_DIR / f"closuregraph_cell10_locked_test_ledger_{CELL10_VERSION}.json"
COMPLETE_PATH = LOG_DIR / f"closuregraph_cell10_locked_test_complete_{CELL10_VERSION}.json"
COMPARISON_FIGURE_PATH = FIGURE_DIR / f"closuregraph_cell10_locked_test_comparison_{CELL10_VERSION}.png"

for directory in (TABLE_DIR, FIGURE_DIR, LOG_DIR, BATCH_DIR):
    directory.mkdir(parents=True, exist_ok=True)

with open(CONFIG_SOURCE, "r", encoding="utf-8") as stream:
    DATA_CONFIG = json.load(stream)
with open(PROTOCOL_SOURCE, "r", encoding="utf-8") as stream:
    CELL9_PROTOCOL = json.load(stream)

CONFIG_SIGNATURE = str(DATA_CONFIG["configuration_sha256"])
GENERATOR_CONFIG = DATA_CONFIG["generator_config"]
DEPTH = int(GENERATOR_CONFIG["sequence_length"])
HEIGHT = int(GENERATOR_CONFIG["image_height"])
IMAGE_WIDTH = int(GENERATOR_CONFIG["image_width"])
PACKED_WIDTH = int(math.ceil(IMAGE_WIDTH / 8.0))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("Cell 10 needs a GPU. Select A100 GPU if available (T4 also works, but slower).")

print(f"Project root       : {PROJECT_ROOT}")
print(f"Dataset version    : {DATASET_VERSION}")
print(f"Frozen run version : {RUN_VERSION}")
print(f"Device             : {DEVICE}")
print(f"GPU                : {torch.cuda.get_device_name(0)}")
print(f"Locked test access : final evaluation only")

if COMPLETE_PATH.exists():
    with open(COMPLETE_PATH, "r", encoding="utf-8") as stream:
        completed = json.load(stream)
    banner("CELL 10 ALREADY COMPLETE: LOCKED TEST WILL NOT BE OPENED AGAIN")
    print(f"Completed at        : {completed.get('completed_utc', 'unknown')}")
    print(f"Test summary        : {SUMMARY_PATH}")
    print(f"Paired significance : {SIGNIFICANCE_PATH}")
    print(f"Sample metrics      : {RECORDS_PATH}")
    print(f"Comparison figure   : {COMPARISON_FIGURE_PATH}")
    raise SystemExit(0)


def checkpoint_path(phase: str, seed: int) -> Path:
    return MODEL_DIR / f"closuregraph_{phase}_{RUN_VERSION}_seed{seed}_best.pt"


checkpoint_manifest = []
for phase, display_name in (("baseline", "Baseline"), ("graph_amplitude", "Graph + amplitude")):
    for seed in TRAINING_SEEDS:
        path = checkpoint_path(phase, seed)
        if not path.exists():
            raise FileNotFoundError(f"Required frozen checkpoint is missing: {path}")
        checkpoint_manifest.append({"phase": phase, "model": display_name, "seed": seed, "path": str(path), "bytes": path.stat().st_size})
show_table(pd.DataFrame(checkpoint_manifest), "Frozen Cell-9R checkpoints", digits=0)

protocol_core = {
    "cell": 10,
    "version": CELL10_VERSION,
    "dataset_version": DATASET_VERSION,
    "dataset_configuration_sha256": CONFIG_SIGNATURE,
    "cell9_protocol_sha256": CELL9_PROTOCOL.get("protocol_sha256"),
    "training_seeds": list(TRAINING_SEEDS),
    "evaluation_models": ["baseline", "graph_amplitude"],
    "test_batch_size": TEST_BATCH_SIZE,
    "prediction_aggregation": "mean of the three same-model independent-seed checkpoints",
    "locked_test_policy": "one final evaluation; per-batch atomic records enable resume without saved-batch repetition",
}
protocol_signature = signature_of(protocol_core)
atomic_json({**protocol_core, "protocol_sha256": protocol_signature}, CONFIG_PATH)


class SeparableConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.spatial_depthwise = nn.Conv3d(in_channels, in_channels, (1, 3, 3), padding=(0, 1, 1), groups=in_channels, bias=False)
        self.temporal_pointwise = nn.Conv3d(in_channels, out_channels, (TEMPORAL_KERNEL, 1, 1), padding=(TEMPORAL_KERNEL // 2, 0, 0), bias=False)
        self.activation = nn.PReLU(num_parameters=out_channels)
        self.normalization = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=DROPOUT_PROBABILITY)
    def forward(self, x):
        return self.dropout(self.normalization(self.activation(self.temporal_pointwise(self.spatial_depthwise(x)))))


class DoubleSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(SeparableConv3d(in_channels, out_channels), SeparableConv3d(out_channels, out_channels))
    def forward(self, x):
        return self.layers(x)


class UNet3D3SM(nn.Module):
    def __init__(self, base_channels: int = BASE_CHANNELS):
        super().__init__()
        c = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.channels = c
        self.pool = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.encoder_1, self.encoder_2 = DoubleSeparableBlock(2, c[0]), DoubleSeparableBlock(c[0], c[1])
        self.encoder_3, self.encoder_4 = DoubleSeparableBlock(c[1], c[2]), DoubleSeparableBlock(c[2], c[3])
        self.bottleneck = DoubleSeparableBlock(c[3], c[4])
        self.up_4, self.decoder_4 = nn.ConvTranspose3d(c[4], c[3], (1,2,2), (1,2,2)), DoubleSeparableBlock(c[3]*2, c[3])
        self.up_3, self.decoder_3 = nn.ConvTranspose3d(c[3], c[2], (1,2,2), (1,2,2)), DoubleSeparableBlock(c[2]*2, c[2])
        self.up_2, self.decoder_2 = nn.ConvTranspose3d(c[2], c[1], (1,2,2), (1,2,2)), DoubleSeparableBlock(c[1]*2, c[1])
        self.up_1, self.decoder_1 = nn.ConvTranspose3d(c[1], c[0], (1,2,2), (1,2,2)), DoubleSeparableBlock(c[0]*2, c[0])
        self.output_layer = nn.Conv3d(c[0], 1, kernel_size=1)
    def decode_features(self, x):
        e1 = self.encoder_1(x); e2 = self.encoder_2(self.pool(e1)); e3 = self.encoder_3(self.pool(e2)); e4 = self.encoder_4(self.pool(e3)); b = self.bottleneck(self.pool(e4))
        d4 = self.decoder_4(torch.cat([self.up_4(b), e4], 1)); d3 = self.decoder_3(torch.cat([self.up_3(d4), e3], 1)); d2 = self.decoder_2(torch.cat([self.up_2(d3), e2], 1))
        return self.decoder_1(torch.cat([self.up_1(d2), e1], 1))
    def forward(self, x):
        return self.output_layer(self.decode_features(x))


class ReliabilityWeightedTemporalGraph(nn.Module):
    def __init__(self, channels: int, lags: tuple[int, ...]):
        super().__init__(); self.lags = tuple(int(v) for v in lags)
        self.message_projections = nn.ModuleList([nn.Sequential(nn.Conv3d(channels, channels, 1, bias=False), nn.PReLU(num_parameters=channels), nn.BatchNorm3d(channels)) for _ in self.lags])
        self.lag_strengths = nn.Parameter(torch.zeros(len(self.lags)))
        self.update = nn.Sequential(nn.Conv3d(channels*2, channels, 1, bias=False), nn.PReLU(num_parameters=channels), nn.BatchNorm3d(channels), nn.Dropout3d(.10), nn.Conv3d(channels, channels, 1, bias=False))
        self.gate = nn.Sequential(nn.Conv3d(channels*2, channels, 1), nn.Sigmoid())
    def forward(self, features, valid_mask):
        aggregate, normalizer = torch.zeros_like(features), torch.zeros_like(valid_mask)
        strengths = F.softplus(self.lag_strengths) + 1e-4
        for i, lag in enumerate(self.lags):
            if lag >= features.shape[2]: continue
            projected, strength = self.message_projections[i](features), strengths[i] / float(lag)
            early, late = valid_mask[:, :, :-lag], valid_mask[:, :, lag:]
            aggregate += F.pad(strength * projected[:, :, :-lag] * early, (0,0,0,0,lag,0)) + F.pad(strength * projected[:, :, lag:] * late, (0,0,0,0,0,lag))
            normalizer += F.pad(strength * early, (0,0,0,0,lag,0)) + F.pad(strength * late, (0,0,0,0,0,lag))
        combined = torch.cat([features, aggregate / normalizer.clamp_min(1e-4) - features], 1)
        return features + self.gate(combined) * self.update(combined)


class GraphAmplitudeInSAR(UNet3D3SM):
    def __init__(self, base_channels: int = BASE_CHANNELS):
        super().__init__(base_channels); self.temporal_graph = ReliabilityWeightedTemporalGraph(self.channels[0], GRAPH_LAGS)
    def forward(self, x):
        return self.output_layer(self.temporal_graph(self.decode_features(x), x[:, 1:2]))


def assert_finite_state(checkpoint: dict, path: Path) -> None:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict): raise RuntimeError(f"No model_state_dict in {path}")
    for name, value in state.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite checkpoint tensor {name}: {path}")


def load_models() -> dict[str, list[nn.Module]]:
    models = {"baseline": [], "graph_amplitude": []}
    for phase in models:
        for seed in TRAINING_SEEDS:
            path = checkpoint_path(phase, seed)
            checkpoint = safe_torch_load(path, DEVICE); assert_finite_state(checkpoint, path)
            model = (UNet3D3SM() if phase == "baseline" else GraphAmplitudeInSAR()).to(DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.eval()
            models[phase].append(model)
    return models


def discover_locked_shards() -> list[Path]:
    paths = sorted(DATA_DIR.glob(f"locked_test_{DATASET_VERSION}_*.h5"))
    if not paths: raise FileNotFoundError(f"No locked_test shards found in {DATA_DIR}")
    expected = int(DATA_CONFIG["split_specifications"]["locked_test"]["count"])
    counts = []
    for path in paths:
        with h5py.File(path, "r") as h:
            if h.attrs.get("status", "") != "complete" or h.attrs.get("split", "") != "locked_test" or not bool(h.attrs.get("locked_for_final_only", False)):
                raise RuntimeError(f"Invalid locked-test policy attributes in {path}")
            if h.attrs.get("configuration_sha256", "") != CONFIG_SIGNATURE: raise RuntimeError(f"Configuration mismatch: {path}")
            n = int(h.attrs["number_of_samples"]); counts.append(n)
            expected_shapes = {"observed_normalized": (n, DEPTH, HEIGHT, IMAGE_WIDTH), "valid_mask_packed": (n, DEPTH, HEIGHT, PACKED_WIDTH), "target_spatial_mm": (n, HEIGHT, IMAGE_WIDTH), "target_temporal_profile": (n, DEPTH)}
            for key, shape in expected_shapes.items():
                if tuple(h[key].shape) != shape: raise RuntimeError(f"Unexpected {key} shape in {path}")
    if sum(counts) != expected: raise RuntimeError(f"Locked-test count is {sum(counts)}, expected {expected}")
    return paths


def batch_from_handle(h: h5py.File, indices: np.ndarray, global_offset: int) -> dict[str, torch.Tensor]:
    observed = h["observed_normalized"][indices].astype(np.float32)
    packed = h["valid_mask_packed"][indices]
    valid = np.unpackbits(packed, axis=-1, count=IMAGE_WIDTH, bitorder="little").astype(np.float32)
    spatial = h["target_spatial_mm"][indices].astype(np.float32)
    temporal = h["target_temporal_profile"][indices].astype(np.float32)
    target_mm = temporal[:, :, None, None] * spatial[:, None, :, :]
    lo = h["metadata/scale_min_mm"][indices].astype(np.float32); hi = h["metadata/scale_max_mm"][indices].astype(np.float32)
    scale = np.maximum(hi - lo, 1e-6)
    target = (2.0 * (target_mm - lo[:, None, None, None]) / scale[:, None, None, None] - 1.0)[:, None].astype(np.float32)
    return {"input": torch.from_numpy(np.stack([observed, valid], 1)), "target": torch.from_numpy(target), "target_mm": torch.from_numpy(target_mm), "scale_min": torch.from_numpy(lo), "scale_range": torch.from_numpy(scale), "sample_seed": torch.from_numpy(h["metadata/sample_seed"][indices].astype(np.int64)), "sample_index": torch.arange(global_offset + int(indices[0]), global_offset + int(indices[-1]) + 1, dtype=torch.long)}


def metric_vectors(pred_norm: torch.Tensor, target_norm: torch.Tensor, target_mm: torch.Tensor, lo: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_mm = ((pred_norm[:, 0] + 1.0) * 0.5 * scale[:, None, None, None] + lo[:, None, None, None])
    error = pred_mm - target_mm
    norm_error = pred_norm - target_norm
    mae = error.abs().mean((1,2,3)); rmse = error.square().mean((1,2,3)).sqrt()
    # norm_error retains the singleton output-channel axis, so reduce all
    # non-batch axes.  The prior reduction left the image-width axis (128
    # values) and failed when a per-sequence scalar was written to CSV.
    mse_norm = norm_error.square().mean((1,2,3,4))
    # SSIM is computed on normalized [-1, 1] frames with a fixed data range of two.
    x, y = pred_norm[:, 0], target_norm[:, 0]
    mux, muy = x.mean((-2,-1)), y.mean((-2,-1))
    vx = (x - mux[...,None,None]).square().mean((-2,-1)); vy = (y - muy[...,None,None]).square().mean((-2,-1)); cov = ((x-mux[...,None,None])*(y-muy[...,None,None])).mean((-2,-1))
    c1, c2 = (0.01*2.0)**2, (0.03*2.0)**2
    ssim = (((2*mux*muy+c1)*(2*cov+c2))/((mux.square()+muy.square()+c1)*(vx+vy+c2))).mean(1)
    snr = target_mm.square().mean((1,2,3)) / error.square().mean((1,2,3)).clamp_min(1e-10)
    peak_bias = (pred_mm.abs().amax((1,2,3)) - target_mm.abs().amax((1,2,3))).abs()
    pt, tt = pred_mm.mean((-2,-1)), target_mm.mean((-2,-1)); pc, tc = pt-pt.mean(1, keepdim=True), tt-tt.mean(1, keepdim=True)
    temporal_corr = (pc*tc).sum(1) / (pc.square().sum(1).sqrt()*tc.square().sum(1).sqrt()).clamp_min(1e-8)
    pred_onset = (pt[:,1:]-pt[:,:-1]).abs().argmax(1); true_onset = (tt[:,1:]-tt[:,:-1]).abs().argmax(1)
    return {"mae_mm": mae, "rmse_mm": rmse, "mse_normalized": mse_norm, "ssim": ssim, "snr_linear": snr, "absolute_peak_bias_mm": peak_bias, "temporal_correlation": temporal_corr, "absolute_onset_error_epochs": (pred_onset-true_onset).abs().float()}


def expected_batch_path(start: int) -> Path:
    return BATCH_DIR / f"batch_{start:06d}.csv"


def batch_is_valid(path: Path, start: int, count: int) -> bool:
    if not path.exists(): return False
    try:
        frame = pd.read_csv(path)
        required = {"sample_index", "model", "seed"}
        return required.issubset(frame.columns) and set(frame["sample_index"].astype(int)) == set(range(start, start+count)) and len(frame) == count * 8
    except Exception:
        return False


def evaluate_locked_test() -> pd.DataFrame:
    stage(4, "Opening and evaluating the locked test exactly once.", "Six frozen checkpoints are evaluated together. A completed batch is atomically saved before the next batch starts.")
    shards = discover_locked_shards()
    shard_counts = []
    for path in shards:
        with h5py.File(path, "r") as h: shard_counts.append(int(h.attrs["number_of_samples"]))
    total = sum(shard_counts)
    existing = sorted(BATCH_DIR.glob("batch_*.csv"))
    atomic_json({"status":"in_progress", "protocol_sha256": protocol_signature, "locked_test_shards":[str(p) for p in shards], "expected_sequences":total, "saved_batch_files":len(existing), "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, LEDGER_PATH)
    models = load_models()
    print(f"Locked test sequences: {total}; saved batches available: {len(existing)}")
    offset = 0; processed = 0
    with torch.inference_mode():
        for shard_path, shard_count in zip(shards, shard_counts):
            with h5py.File(shard_path, "r", swmr=True) as h:
                for local_start in range(0, shard_count, TEST_BATCH_SIZE):
                    n = min(TEST_BATCH_SIZE, shard_count-local_start); global_start = offset + local_start; output_path = expected_batch_path(global_start)
                    if batch_is_valid(output_path, global_start, n):
                        processed += n; continue
                    local_indices = np.arange(local_start, local_start+n)
                    batch = batch_from_handle(h, local_indices, offset)
                    inp = batch["input"].to(DEVICE, non_blocking=True); target = batch["target"].to(DEVICE); target_mm = batch["target_mm"].to(DEVICE); lo = batch["scale_min"].to(DEVICE); scale = batch["scale_range"].to(DEVICE)
                    predictions = {phase: [] for phase in models}
                    for phase, phase_models in models.items():
                        for model in phase_models:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): out = model(inp)
                            if not torch.isfinite(out).all(): raise FloatingPointError(f"Non-finite locked-test prediction for {phase}, batch {global_start}")
                            predictions[phase].append(out.float())
                    rows = []
                    indexes, seeds = batch["sample_index"].tolist(), batch["sample_seed"].tolist()
                    for phase, name in (("baseline", "Baseline"), ("graph_amplitude", "Graph + amplitude")):
                        for seed, prediction in zip(TRAINING_SEEDS, predictions[phase]):
                            vectors = metric_vectors(prediction, target, target_mm, lo, scale)
                            for i in range(n): rows.append({"sample_index":indexes[i], "sample_seed":seeds[i], "model":name, "seed":int(seed), "aggregation":"single_checkpoint", **{key:float(value[i].item()) for key,value in vectors.items()}})
                        ensemble = torch.stack(predictions[phase]).mean(0)
                        vectors = metric_vectors(ensemble, target, target_mm, lo, scale)
                        for i in range(n): rows.append({"sample_index":indexes[i], "sample_seed":seeds[i], "model":name, "seed":"ensemble", "aggregation":"three_seed_mean_prediction", **{key:float(value[i].item()) for key,value in vectors.items()}})
                    atomic_csv(pd.DataFrame(rows), output_path)
                    processed += n
                    if processed % 50 == 0 or processed == total: print(f"Locked-test evaluation saved: {processed}/{total} sequences")
            offset += shard_count
    frames = [pd.read_csv(expected_batch_path(start)) for start in range(0, total, TEST_BATCH_SIZE) if expected_batch_path(start).exists()]
    records = pd.concat(frames, ignore_index=True)
    expected_rows = total * 8
    if len(records) != expected_rows or records["sample_index"].nunique() != total:
        raise RuntimeError(f"Incomplete locked-test record set: rows={len(records)}, expected={expected_rows}")
    atomic_csv(records.sort_values(["sample_index","model","aggregation","seed"]), RECORDS_PATH)
    atomic_json({"status":"evaluation_complete", "protocol_sha256":protocol_signature, "locked_test_sequences":total, "record_rows":int(len(records)), "completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, LEDGER_PATH)
    return records


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64); n = len(values)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 200):
        take = min(200, BOOTSTRAP_REPLICATES-start); means[start:start+take] = values[rng.integers(0,n,size=(take,n))].mean(1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values); order = np.argsort(p_values); adjusted = np.empty(m); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m-rank)*p_values[index])); adjusted[index] = running
    return adjusted.tolist()


METRICS = [("mae_mm", "lower"), ("rmse_mm", "lower"), ("mse_normalized", "lower"), ("ssim", "higher"), ("snr_linear", "higher"), ("absolute_peak_bias_mm", "lower"), ("temporal_correlation", "higher"), ("absolute_onset_error_epochs", "lower")]


def analyse(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage(5, "Summarising the frozen locked-test results and paired differences.", "All comparisons use the preselected three-seed mean prediction; the test set is never used for checkpoint selection.")
    ensemble = records[records["aggregation"] == "three_seed_mean_prediction"].copy()
    summary_rows = []
    for model, group in ensemble.groupby("model", sort=False):
        row = {"model":model, "aggregation":"three_seed_mean_prediction", "n_sequences":int(group["sample_index"].nunique())}
        for metric, _ in METRICS: row[metric] = float(group[metric].mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows); atomic_csv(summary, SUMMARY_PATH)
    seed_rows = []
    for (model, seed), group in records[records["aggregation"] == "single_checkpoint"].groupby(["model","seed"], sort=False):
        row = {"model":model, "seed":int(seed), "n_sequences":int(group["sample_index"].nunique())}
        for metric, _ in METRICS: row[metric] = float(group[metric].mean())
        seed_rows.append(row)
    seed_summary = pd.DataFrame(seed_rows); atomic_csv(seed_summary, SEED_SUMMARY_PATH)
    base = ensemble[ensemble["model"] == "Baseline"].set_index("sample_index").sort_index(); graph = ensemble[ensemble["model"] == "Graph + amplitude"].set_index("sample_index").sort_index()
    if not base.index.equals(graph.index): raise RuntimeError("Paired locked-test sequence indexes do not match.")
    rng = np.random.default_rng(10010); rows=[]; p=[]
    for metric, direction in METRICS:
        improvement = (base[metric].to_numpy()-graph[metric].to_numpy()) if direction == "lower" else (graph[metric].to_numpy()-base[metric].to_numpy())
        low, high = bootstrap_ci(improvement, rng)
        if wilcoxon is None: p_value = float("nan")
        else:
            try: p_value = float(wilcoxon(improvement, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
            except ValueError: p_value = 1.0
        p.append(p_value)
        rows.append({"comparison":"graph_amplitude_vs_baseline", "metric":metric, "direction":direction, "n_pairs":len(improvement), "mean_oriented_improvement":float(improvement.mean()), "bootstrap_ci_95_low":low, "bootstrap_ci_95_high":high, "wilcoxon_p_value":p_value})
    finite = [value if math.isfinite(value) else 1.0 for value in p]; corrected = holm_adjust(finite)
    for row, value in zip(rows, corrected): row["p_holm"] = value; row["significant_holm_0_05"] = bool(value < .05)
    significance = pd.DataFrame(rows); atomic_csv(significance, SIGNIFICANCE_PATH)
    show_table(summary, "Locked-test ensemble summary")
    show_table(significance, "Paired locked-test significance")
    return summary, seed_summary, significance


def draw_figure(summary: pd.DataFrame) -> None:
    stage(6, "Creating the final locked-test comparison figure.", "This figure is descriptive only; all checkpoints were frozen before opening the final test set.")
    metrics = METRICS; fig, axes = plt.subplots(2, 4, figsize=(18, 8)); colors={"Baseline":"#777777", "Graph + amplitude":"#009E73"}
    for axis, (metric, direction) in zip(axes.flat, metrics):
        ordered = summary.set_index("model").loc[["Baseline","Graph + amplitude"]]
        bars = axis.bar(ordered.index, ordered[metric], color=[colors[v] for v in ordered.index])
        axis.set_title(metric.replace("_", " ")); axis.set_ylabel("lower is better" if direction=="lower" else "higher is better")
        for bar, value in zip(bars, ordered[metric]): axis.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{value:.4g}", ha="center", va="bottom", fontsize=9)
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("ClosureGraph-InSAR locked-test evaluation (three-seed frozen ensembles)", fontsize=16, fontweight="bold")
    fig.tight_layout(); save_figure_atomic(fig, COMPARISON_FIGURE_PATH); plt.show(); plt.close(fig)


if LEDGER_PATH.exists() and json.load(open(LEDGER_PATH, "r", encoding="utf-8")).get("status") == "evaluation_complete" and RECORDS_PATH.exists():
    print("Locked-test batch records already complete; rebuilding final tables without reopening locked data.")
    records_frame = pd.read_csv(RECORDS_PATH)
else:
    records_frame = evaluate_locked_test()

summary_frame, seed_frame, significance_frame = analyse(records_frame)
draw_figure(summary_frame)

stage(7, "Writing the final locked-test audit.", "The final locked-test set has now been evaluated once. Do not rerun this cell; later runs will only display the saved outputs.")
atomic_json({"status":"complete", "protocol_sha256":protocol_signature, "completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "locked_test_sequences":int(summary_frame["n_sequences"].iloc[0]), "summary":str(SUMMARY_PATH), "seed_summary":str(SEED_SUMMARY_PATH), "significance":str(SIGNIFICANCE_PATH), "records":str(RECORDS_PATH), "figure":str(COMPARISON_FIGURE_PATH)}, COMPLETE_PATH)
banner("CELL 10 COMPLETE: FINAL LOCKED-TEST EVALUATION SAVED")
print(f"Summary table       : {SUMMARY_PATH}")
print(f"Seed summary        : {SEED_SUMMARY_PATH}")
print(f"Significance table  : {SIGNIFICANCE_PATH}")
print(f"Sample metrics      : {RECORDS_PATH}")
print(f"Comparison figure   : {COMPARISON_FIGURE_PATH}")
print(f"Final audit         : {COMPLETE_PATH}")
