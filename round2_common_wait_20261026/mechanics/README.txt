Paper 3 round-two common-wait verification data

18 trajectories: six conditions paired within three independent preparations.
Translation begins at 161.080 ps after 4 ps equilibration and 157.080 ps pretranslation evolution.
series/: exact force traces and original metadata; trajectories.json maps all 18.
Columns: time_ps, travel_nm, Ft_nN (positive resistance along +x), Fz_nN
(signed tool reaction), z_A, theta_rad, vz_A_ps. Pz=-Fz*vz; input=.03*Ft+Pz
in microW. The analysis uses 4--16 nm; 6--16 and 8--16 nm check sensitivity.
source_logs/: exact original logs, including both A10B20-R01 segments. The
selected CSV excludes duplicate/restart initialization rows as documented in metadata.
Original absolute paths are provenance only, never execution dependencies.

Install requirements.txt in your Python environment. From the extracted directory:
  python replay_round2_data.py
This verifies every supplied file hash, replays 18 force histories, 54 window
records, quadrature/downsampling and independent Gaussian checks, and writes
replay_result.json, including the Fig.6 means/sample SDs. No MD is launched.
Results preserve 14/15 lower-force pairs and 15/15 higher-input pairs; A10B05-R01
increases force by 3.30%. Windows of a trajectory are not independent samples.

primary_reference.py and reduce_new_md.py supply the unchanged numerical kernels.
Other analysis_code scripts are verbatim provenance and retain repository paths;
do not run them directly in this package. This is a verification subset, not the
full MD campaign, restart archive or executable simulation environment.
The original two-preparation public release is a separate historical dataset.

Current structural coverage: Supplementary Data2 contains18 stations in three preparations,40 R02 phase configurations and all three relaxed references.

A25B10 R01 now uses the retained S4_R01_recovery continuation for both forces and structure. All other17 histories retain their previous values. Its source log contains the initial prefix plus continuation from575000; five obsolete pre-restart rows are excluded as recorded in metadata.

controls/ contains five adopted sensitivity controls and two original-ramp unit-scale A25B20 comparisons. They retain their own matched histories and do not enter the18 primary common-wait trajectories. Run python replay_sensitivity_controls.py separately to reproduce their reported values.
