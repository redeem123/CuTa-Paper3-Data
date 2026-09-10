# Release validation

Validated locally on 2026-09-10 in a clean copy of the exported repository.

The primary force samples were rebuilt from included LAMMPS logs using `audit_provenance.py --rebuild-samples`. The offline scalar/plot workflow and supplementary numerical checks completed. All six tests listed in the README passed, including analytic force/work checks, power identities, common-window statistics and included atomistic source hashes.

Rebuilt `data/run_metrics.csv` matched the manuscript dataset byte for byte. Its SHA-256 is `b0c3fec43bb13d9f6964ac1807b33229d0e322a45140e23991c42b1ec0474f66`.

All 138 files in `data_manifest.json` matched their export hashes. No full simulation rerun or fresh OVITO rendering was performed during this release check. Archived PTM/RDF summaries and model illustrations remain retained evidence, not newly reproduced structural calculations.
