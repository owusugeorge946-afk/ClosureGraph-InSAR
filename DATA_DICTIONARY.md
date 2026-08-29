# Data dictionary

The training and evaluation scripts expect an HDF5-based production dataset. The underlying arrays are not included in this GitHub preparation package.

| Field | Meaning | Expected form |
|---|---|---|
| Observed LOS sequence | Incomplete normalized LOS displacement time series | 20 acquisitions × 128 × 128 pixels |
| Validity mask | Indicates observed/valid values in the LOS sequence | Same spatial-temporal support as the observed sequence |
| Reference target | Complete displacement sequence used for supervised learning and evaluation | 20 acquisitions × 128 × 128 pixels |
| Split metadata | Train, validation, and locked-test allocation | JSON configuration with checksum |
| Sample metadata | Sequence identifier and event descriptors used in saved results | CSV/JSON sidecar metadata |

The repository should contain no participant data, location-sensitive data, credentials, or unredacted Google Drive paths.
