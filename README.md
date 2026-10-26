# Cu-Ta nanoscratching data and analysis

## Accepted-paper common-wait release

The article by Thanh-Huan Nguyen, Viet-Anh Ngo and Duc-Toan Nguyen has been accepted in Materials Today Communications, production reference MTCOMM 116493.

Use [paper3-v2.0-common-wait](https://github.com/redeem123/CuTa-Paper3-Data/releases/tag/paper3-v2.0-common-wait) for the accepted-paper dataset. It contains eighteen primary common-wait force histories, eighteen loaded structural stations across all six conditions in R01–R03, forty phase-sampled R02 configurations and three relaxed references. Full datasets are release ZIP assets. Versioned code, replay instructions, checksums and licensing are in [round2_common_wait_20261026](round2_common_wait_20261026/).

Original v2.0.0 data are CC-BY 4.0 and original Python code is MIT. Third-party potential terms are unchanged. These grants apply to the new version only. A Zenodo archival deposit is in preparation. Its DOI will be added after publication.

## Historical v1.0 artifact

The [paper3-v1.0 release](https://github.com/redeem123/CuTa-Paper3-Data/releases/tag/paper3-v1.0) and its tag are preserved. Its two-preparation histories are separate from the accepted-paper common-wait dataset. The following instructions apply to that historical artifact.


Data and analysis code supporting the manuscript **Deformation and force–power trade-offs in vibration-assisted nanoscratching of Cu–Ta amorphous nanolaminates** by Thanh-Huan Nguyen, Viet-Anh Ngo, and Duc-Toan Nguyen.

Corresponding author: Duc-Toan Nguyen, toan.nguyenduc@sis.hust.edu.vn.

The sections below describe the historical two-preparation v1.0 artifact. The accepted-paper common-wait release is linked above. The companion Paper 1 is still in preparation. Shared atomistic models and simulation evidence are identified through provenance records; this repository supports Paper 3's stroke-resolved force and mechanical input analysis.

## What is included

- Twelve primary run logs and launch records, covering a baseline and five vibration conditions in two realizations.
- CSV force samples, including five control runs, and JSON metadata.
- Derived force, mechanical power, cycle-work, window-sensitivity and quadrature results.
- Twelve reduced atomistic configurations for qualitative morphology and strain views, with original atom IDs and coordinates.
- Analysis and rendering scripts, figure assets and editable generated table fragments.
- SHA-256 hashes in `data_manifest.json` and run-specific provenance records.

No manuscript, cover letter, journal forms, or private Git history is included. Complete atomistic trajectories are not included. Original machine paths in provenance metadata identify archived source locations and are not required for an offline rebuild. Do not use the scripts' archive/import modes without access to the original simulation volume.

## Reproduce numerical results

Use Python 3 with NumPy 2+, SciPy and Matplotlib. Rendering requires OVITO and its additional dependencies. Quantitative plot generation also requires a working LaTeX installation because the plotting code uses TeX text rendering.

```sh
python -m pip install -r requirements.txt
python audit_provenance.py --rebuild-samples
python rebuild_analysis.py --offline
python generate_supplementary_data.py
python -m unittest test_analysis.py \
  test_reproducibility.ReproducibilityTests.test_all_power_identities \
  test_reproducibility.ReproducibilityTests.test_common_window_summary \
  test_reproducibility.ReproducibilityTests.test_visualization_sources
```

The manuscript-text regression test in the copied `test_reproducibility.py` belongs to the private manuscript build and is not part of the public command above. It requires manuscript source files that are intentionally absent here.

For optional reconstruction of the atomistic panels, install OVITO in a suitable environment and run:

```sh
QT_QPA_PLATFORM=offscreen python render_audited_views.py
```

The twelve primary force series can be checked against the archived executed logs. Controls are rebuilt from packaged samples, not archived control logs. The RDF arrays, PTM summaries, and model illustrations are retained outputs and are not recomputed by the offline numerical workflow. Final atomistic snapshots are at different travel distances outside the common 4–16 nm force window. They do not independently establish the mechanism of the window-averaged force response. See `FIGURE_PROVENANCE.md`.

## Version and citation

Use release `paper3-v1.0` only when referring to the original two-preparation submission evidence. Cite the authors, repository title, release version, year 2026 and the release URL. The historical v1.0 tag retains its original citation metadata. The current root `CITATION.cff` describes version 2.0.0.

Public availability does not by itself assign a permissive reuse license. No additional license grant is asserted in this release. Contact the corresponding author for reuse permissions beyond those otherwise available.

## AI assistance

OpenAI Codex and Google Gemini assisted with analysis/visualization code and manuscript preparation. Scientific provenance and numerical checks are provided to support independent verification. This statement is not a substitute for author verification.
