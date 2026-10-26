# Accepted-paper common-wait data and code

This version supports **Deformation and force–power trade-offs in vibration-assisted nanoscratching of Cu–Ta amorphous nanolaminates**, by Thanh-Huan Nguyen, Viet-Anh Ngo and Duc-Toan Nguyen, accepted in Materials Today Communications (production reference MTCOMM 116493).

The present evidence covers eighteen primary force histories and eighteen loaded structural stations: six conditions paired within three independent preparations, R01–R03. Forty additional phase configurations belong to R02 and are correlated observations. Sensitivity controls retain separately identified histories. The original two-preparation `paper3-v1.0` release remains a historical dataset; do not mix its A10 histories with the present common-wait comparisons.

## Obtain the complete datasets

Download the two complete data archives from [release paper3-v2.0-common-wait](https://github.com/redeem123/CuTa-Paper3-Data/releases/tag/paper3-v2.0-common-wait). A Zenodo archival deposit is being prepared and will be linked when published. `DATASET_MANIFEST.json` records their exact filenames, byte counts and SHA-256 values. The `.py` files shown here are byte-identical copies from those archives, published for code inspection. This Git directory alone does not contain the coordinate/force datasets needed for replay.

Extract each ZIP into its own new directory and follow its `README.txt`. Do not combine the mechanics and structure ZIP contents in one directory.

- Data 1, `supplementary_data_mechanics.zip`: eighteen force histories, source logs, preparation/run metadata, reported reductions, portable mechanical replay and separate sensitivity replay.
- Data 2, `supplementary_data_structure.zip`: eighteen station configurations, forty phase configurations, three relaxed references, retained potential, structural/contact results, original verification logs and portable replay.

From the extracted Data 1 directory:

```sh
python replay_round2_data.py
python replay_sensitivity_controls.py
```

From the extracted Data 2 directory, with its documented Python dependencies and `zstd` installed:

```sh
python replay_structure_data.py --work /absolute/path/to/new/replay_directory --workers 3
```

These default commands analyze existing records and launch no MD. Provenance-only producer scripts under `analysis_code/` retain original repository paths and should not be executed directly from this Git checkout.

## Evidence scope

The common-wait translation origin is 161.080 ps and the principal mechanical comparison uses 4–16 nm travel. The paired force comparison retains fourteen reductions out of fifteen, including the adverse 3.30% increase for A10B05–R01; all fifteen pairs increase mechanical input. The A25B10–R01 force and structural data use the same qualified retained recovery history. The data packages preserve preparation identities, overlap selection, exact loaded-frame travel/phase, source logs and qualification records.

The structural records describe loaded configurations, not permanent post-unloading damage. The A25B10/A50B20 amplitude pair matches travel and phase in all three preparations; comparisons across frequencies report their actual phases. The package is a selected verification dataset, not a full production-trajectory or simulation-executable archive. Existing numerical qualification receipts are preserved; no new simulation or experimental validation is claimed by this release.

## Licenses

Original data, metadata and documentation in this version are licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original Python code is licensed under [MIT](LICENSE-CODE-MIT.txt). The retained third-party potential keeps its original attribution and terms. See [LICENSES_AND_CHECKSUMS.md](LICENSES_AND_CHECKSUMS.md) for the exact scope and archive identities. These new grants do not relicense the historical v1.0 release.
