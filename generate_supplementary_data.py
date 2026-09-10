#!/usr/bin/env python3
"""
generate_supplementary_data.py

Generates structured supplementary tables and audit files for:
- S1: Run and data manifest
- S2: Structural checks (PTM cutoffs and cluster statistics)
- S3: Analysis window sensitivity (4-16, 6-16, 8-16 nm)
- S4: Numerical quadrature, downsampling, and work-power audit
"""

import json
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
GEN_DIR = ROOT / 'generated'
DATA_DIR.mkdir(exist_ok=True)
GEN_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------------
# S1: Run and Data Manifest
# -------------------------------------------------------------------------
def generate_s1_manifest():
    from audit_provenance import run
    run()
    (DATA_DIR / 'supp_manifest.csv').write_bytes((DATA_DIR / 'run_provenance.csv').read_bytes())


def generate_s2_structural():
    ptm_cutoffs = [
        {"rmsd_cutoff": 0.08, "r1_other": 99.983, "r1_fcc": 0.005, "r1_hcp": 0.011, "r1_bcc": 0.000, "r1_cryst": 0.017,
                              "r2_other": 99.983, "r2_fcc": 0.006, "r2_hcp": 0.010, "r2_bcc": 0.001, "r2_cryst": 0.017},
        {"rmsd_cutoff": 0.10, "r1_other": 99.884, "r1_fcc": 0.031, "r1_hcp": 0.078, "r1_bcc": 0.008, "r1_cryst": 0.116,
                              "r2_other": 99.897, "r2_fcc": 0.030, "r2_hcp": 0.067, "r2_bcc": 0.007, "r2_cryst": 0.103},
        {"rmsd_cutoff": 0.12, "r1_other": 99.490, "r1_fcc": 0.107, "r1_hcp": 0.348, "r1_bcc": 0.055, "r1_cryst": 0.510,
                              "r2_other": 99.471, "r2_fcc": 0.109, "r2_hcp": 0.365, "r2_bcc": 0.055, "r2_cryst": 0.529},
        {"rmsd_cutoff": 0.15, "r1_other": 96.959, "r1_fcc": 0.583, "r1_hcp": 2.115, "r1_bcc": 0.343, "r1_cryst": 3.041,
                              "r2_other": 96.847, "r2_fcc": 0.601, "r2_hcp": 2.201, "r2_bcc": 0.352, "r2_cryst": 3.153},
    ]
    out_csv = DATA_DIR / 'supp_ptm_cutoffs.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(ptm_cutoffs[0].keys()))
        writer.writeheader()
        writer.writerows(ptm_cutoffs)
    print(f'Wrote {out_csv}')

    cluster_stats = {
        "cutoff_A": 3.2,
        "rmsd_cutoff": 0.10,
        "realization_1": {
            "n_atoms_total": 543778,
            "cryst_matches_total": 633,
            "cryst_fraction_pct": 0.116,
            "total_clusters": 457,
            "single_atom_clusters": 363,
            "single_atom_fraction_pct": 79.4,
            "two_atom_clusters": 67,
            "three_atom_clusters": 15,
            "four_plus_clusters": 12,
            "max_cluster_size_atoms": 12,
            "max_cluster_volume_nm3": "< 0.20"
        },
        "realization_2": {
            "n_atoms_total": 544324,
            "cryst_matches_total": 563,
            "cryst_fraction_pct": 0.103,
            "total_clusters": 425,
            "single_atom_clusters": 348,
            "single_atom_fraction_pct": 81.9,
            "two_atom_clusters": 53,
            "three_atom_clusters": 13,
            "four_plus_clusters": 11,
            "max_cluster_size_atoms": 23,
            "max_cluster_volume_nm3": "< 0.35"
        }
    }
    out_json = DATA_DIR / 'supp_ptm_clusters.json'
    with open(out_json, 'w') as f:
        json.dump(cluster_stats, f, indent=2)
    print(f'Wrote {out_json}')

# -------------------------------------------------------------------------
# S3: Window Sensitivity Audit
# -------------------------------------------------------------------------
def generate_s3_windows():
    from rebuild_supplementary_checks import windows
    windows()


def generate_s4_quadrature():
    from rebuild_supplementary_checks import quadrature
    quadrature()

# -------------------------------------------------------------------------
# S5: Work-Power Audit Table (Single Source of Truth from run_metrics.csv)
# -------------------------------------------------------------------------
def generate_s5_work_power():
    metrics_path = DATA_DIR / 'run_metrics.csv'
    with open(metrics_path, 'r') as f:
        rows = list(csv.DictReader(f))

    base_r1_pin = float([r for r in rows if r['case'] == 'scratch_base_R01'][0]['total_power_uW'])
    base_r2_pin = float([r for r in rows if r['case'] == 'scratch_base_R02'][0]['total_power_uW'])
    base_pin = {1: base_r1_pin, 2: base_r2_pin}

    c_ev = 6.241509126 * 400.0  # eV conversion factor for 400 ps window

    label_map = {
        'scratch_base_R01': (r'Base\_R01', 'Baseline', 'R1'),
        'scratch_base_R02': (r'Base\_R02', 'Baseline', 'R2'),
        'S1_R01': (r'S1\_R01', r'A10B05 ($S_1$)', 'R1'),
        'S1_R02': (r'S1\_R02', r'A10B05 ($S_1$)', 'R2'),
        'S2_R01': (r'S2\_R01', r'A10B20 ($S_2$)', 'R1'),
        'S2_R02': (r'S2\_R02', r'A10B20 ($S_2$)', 'R2'),
        'S4_R01': (r'S4\_R01', r'A25B10 ($S_3$)', 'R1'),
        'S4_R02': (r'S4\_R02', r'A25B10 ($S_3$)', 'R2'),
        'S5_R01': (r'S5\_R01', r'A25B20 ($S_4$)', 'R1'),
        'S5_R02': (r'S5\_R02', r'A25B20 ($S_4$)', 'R2'),
        'S6_R01': (r'S6\_R01', r'A50B20 ($S_5$)', 'R1'),
        'S6_R02': (r'S6\_R02', r'A50B20 ($S_5$)', 'R2'),
    }

    out_rows = []
    tex_lines = []

    for r in rows:
        case = r['case']
        rep = int(r['realization'])
        code, cond, r_str = label_map[case]
        ft = float(r['Ft_mean_nN'])
        px = 0.03 * ft
        pz = float(r['Pz_uW'])
        pin = float(r['total_power_uW'])
        import math
        duration = float(r['window_end_ps']) - float(r['window_start_ps'])
        if not math.isclose(duration, 400.0, abs_tol=1e-9):
            raise ValueError(f'{case}: unexpected evaluation duration {duration}')
        if not math.isclose(pin, px + pz, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'{case}: inconsistent work--power components')
        win = c_ev * pin
        norm = pin / base_pin[rep]

        out_rows.append({
            'case_code': case,
            'condition': r['condition'],
            'realization': f'R{rep}',
            'Ft_mean_nN': ft,
            'Px_uW': px,
            'Pz_uW': pz,
            'Pin_uW': pin,
            'Win_eV': win,
            'norm_Pin': norm
        })

        tex_line = (f"{code:9s} & {cond:14s} & {r_str:6s} & "
                    f"{ft:5.2f}                 & {px:.4f}         & {pz:.4f}         & "
                    f"{pin:.4f}                     & {win:6.1f}                 & {norm:.4f}\\\\")
        tex_lines.append(tex_line)

    out_csv = DATA_DIR / 'supp_work_power.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f'Wrote {out_csv}')

    out_tex = GEN_DIR / 'supp_work_power_rows.tex'
    out_tex.write_text('\n'.join(tex_lines) + '\n' + r'\bottomrule' + '\n')
    print(f'Wrote {out_tex}')

if __name__ == '__main__':
    generate_s1_manifest()
    generate_s2_structural()
    generate_s3_windows()
    generate_s4_quadrature()
    generate_s5_work_power()
    print("All supplementary data generated successfully.")
