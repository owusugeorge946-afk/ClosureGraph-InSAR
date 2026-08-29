# GitHub upload steps

1. Create a new GitHub repository named `ClosureGraph-InSAR`.
2. Upload the contents of this folder, not the folder itself, so that `README.md` appears at the repository root.
3. Before making the repository public, replace every `ADD_PERMANENT_DOI_OR_URL...` value in `metadata/release_links.json`.
4. Upload configuration files and final CSV/JSON results. Use Git LFS or a Zenodo/OSF archive for HDF5 data and PyTorch checkpoint files that exceed GitHub's size limits.
5. Confirm that the repository does not contain Google Drive links with edit permissions, API keys, service-account files, or unpublished test-set experiments.
6. Add a release tag such as `v1.0.0`, archive that release in Zenodo, then paste the Zenodo DOI into `metadata/release_links.json` and the README.

The package is intentionally prepared so the public repository can document the study now while the large data and model files are released through a stable archive.
