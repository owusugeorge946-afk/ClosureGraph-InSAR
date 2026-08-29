# Reproducibility protocol

## Scope

The study evaluates reconstruction of synthetic multitemporal LOS displacement sequences. It does not establish operational performance on real SAR time series. Any downstream use should preserve this limitation.

## Data partition and test protection

The production dataset contains 20,000 training sequences, 2,000 validation sequences, and 2,000 locked-test sequences. Each sequence contains 20 acquisitions on a 128 × 128 grid. The validation split is used for checkpoint selection. The locked-test split is opened only after the model checkpoints are frozen.

`closuregraph_insar_cell10.py` records a completion marker after the locked-test evaluation. If the marker exists, the script stops rather than reopening the test data. Do not delete that marker to obtain new test results.

## Required preconditions

Before running the scripts, confirm that the following are available:

1. The Cell-8 generated HDF5 split files and `configs/production_dataset_config_v1.json`.
2. `configs/closuregraph_cell9_repair_protocol_v2.json`.
3. Six frozen checkpoints: three matched Baseline and three Graph + amplitude checkpoints, one for each seed.
4. A CUDA-enabled Colab runtime for the final evaluation.

Exact filenames and paths are listed in `metadata/expected_artifacts.csv`.

## Execution order

Run Cell 9R to train each model and write frozen checkpoints. Run Cell 10 once to produce per-sequence records, summary tables, paired-significance results, seed summaries, and a completion ledger. Run Cell 11 only after Cell 10; it reads saved CSV/JSON outputs and does not train models or read locked-test shards.

## Release checklist

- Preserve the data split and checkpoint-selection policy.
- Include the configuration JSON files and checksums.
- Release only frozen checkpoints used for the reported results.
- Include all final CSV, JSON, PNG, PDF, and SVG assets referenced by the manuscript.
- Add a permanent archive DOI/URL to `metadata/release_links.json`.
- State clearly whether the public archive contains the locked-test sequences or only final derived outputs.
