# ClosureGraph-InSAR

This repository accompanies the manuscript *ClosureGraph-InSAR: Reliability-Weighted Temporal Graph Refinement for Incomplete Multitemporal Deformation Reconstruction*.

ClosureGraph-InSAR reconstructs incomplete synthetic multitemporal line-of-sight (LOS) displacement sequences using a five-level separable 3D U-Net and a reliability-weighted temporal graph-refinement module. The proposed model uses temporal lags `{1, 2, 4, 8}`. The matched baseline uses the same inputs, backbone, training protocol, and checkpoint-selection procedure, but does not include graph refinement.

## What is included

- Colab scripts for model training, one-time locked-test evaluation, and manuscript-asset generation.
- A machine-readable protocol and a manifest of the data, models, and derived outputs needed for full replication.
- The final architecture figure and the code used to generate it.

## What must be added before public release

The synthetic HDF5 data, frozen checkpoints, and saved CSV/JSON outputs are **not present in this package**. Add them to the paths listed in `metadata/expected_artifacts.csv`, or archive them in Zenodo/OSF and add their permanent DOI or URL to `metadata/release_links.json`.

Do not commit the 2,000-sequence locked-test shard to a public repository before the paper's final evaluation and audit are complete. The public release should include a read-only final-results package rather than a script that permits further test-set experimentation.

## Quick start in Google Colab

1. Create a Google Drive folder named `ClosureGraph_InSAR`.
2. Upload the contents of this repository to that folder, keeping the directory structure.
3. Place the generated data, frozen checkpoints, and saved outputs in the required paths listed in `metadata/expected_artifacts.csv`.
4. In Colab, mount Google Drive and run the scripts in this order:

   - `scripts/closuregraph_insar_cell9_repair_v2.py` — train the three-seed matched Baseline and Graph + amplitude models.
   - `scripts/closuregraph_insar_cell10.py` — run the one-time locked-test evaluation after all checkpoints have been frozen.
   - `scripts/closuregraph_insar_cell11_manuscript_assets.py` — generate figures and tables from saved outputs only.

The scripts were written for a CUDA GPU. Cell 10 explicitly requires a GPU because it evaluates the frozen ensembles.

## Experimental design

| Item | Specification |
|---|---|
| Data | Synthetic LOS displacement sequences with a validity mask |
| Sequence size | 20 acquisitions on a 128 × 128 grid |
| Data split | 20,000 training; 2,000 validation; 2,000 locked-test sequences |
| Models | Matched Baseline; Graph + amplitude |
| Seeds | 31415, 27182, 16180 |
| Graph lags | 1, 2, 4, 8 acquisitions |
| Checkpoint selection | Validation common score only |
| Final test policy | Opened once after checkpoint freezing |

## Reproducibility and reporting

Read `REPRODUCIBILITY.md` before running or releasing the package. It explains the data split, protected-test policy, required files, and expected outputs.

## Citation

Please add the final article citation and DOI after publication. Until then, cite the associated manuscript and this repository version.

## Contact

George Owusu Amoah  
Department of Geography and Regional Planning, University of Cape Coast, Cape Coast, Ghana  
Email: george.amoah003@stu.ucc.edu.gh
