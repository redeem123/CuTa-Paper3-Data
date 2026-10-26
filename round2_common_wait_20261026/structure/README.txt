Paper 3: common-wait structure and contact data

18 structural stations: six conditions in each of R01, R02 and R03.
40 phase configurations: eight samples per oscillatory condition in R02.
All58 exact configurations, three relaxed references, metadata, potential,
reported results and independent checks are supplied. No MD was rerun.

Each station is the last saved frame at/before16nm within4--16nm, selected
before its structural outcomes. Actual travel spans15.8676--15.9807nm and
actual phase is reported. A25B10/A50B20 match travel, frequency, phase and
waiting time in all three preparations. Phase samples belong to R02 only.
The18 stations represent three preparations; the40 phase samples are correlated.

A25B10 R01 uses the retained S4_R01_recovery continuation from575000 for
both force and structural evidence. The initial launcher remains interrupted;
the completed continuation and obsolete-tail selection are documented. Its
mechanical history is supplied in Supplementary Data1. Previous source cohorts
remain preserved and are not mixed into the revised selected histories.

Install requirements.txt in a Python3.11-compatible environment and install zstd.
From the extracted directory run:
  python replay_structure_data.py --work /absolute/path/to/fresh_replay --workers 6
The wrapper verifies all hashes, resolves supplied paths only, recomputes all
18 stations and40 phases, checks independent local fits and compares results.
The work directory must be new. Read replay_result.json and replay.log there.
Default replay does not run LAMMPS. Eighteen original run0 checks are supplied
in results/qualification.json and verification_logs/. To repeat zero-step force
checks, add --engine /absolute/path/to/compatible/lmp. No integration occurs.

D2min is the unnormalized squared affine-fit residual sum in Angstrom^2.
Cutoff3.375A; y periodic only; no affine cell mapping. The reference ROI is
x390--440A,y33--123A,depth5--50A below the original top. Mean and median
remain distinct. Surface heights use bulk-connected metal-atom centres in5A
bins. Contact counts are distinct atoms, not pairs or calibrated contact area.
These loaded configurations do not measure permanent post-unloading damage.

The portable reducer and kernel implement the replay. Other analysis_code
scripts retain repository paths as provenance and should not be run directly
from this export. Original absolute paths in results are historical provenance,
not execution dependencies. Full MD trajectories are outside this selected subset.
