#!/usr/bin/env python3
"""Archive executed evidence and audit the clocks used by the scalar analysis."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import shutil
import numpy as np
from rebuild_analysis import raw_blocks, CASES, interval

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(archive=False, rebuild_samples=False):
    evidence = ROOT / 'evidence'
    rows = []
    segments = {}
    for label, stem, amp, beta in CASES:
        for rep in (1, 2):
            case = f'{stem}_R{rep:02d}'
            meta = json.loads((ROOT / 'data' / f'{case}.json').read_text())
            folder = evidence / case
            if archive:
                folder.mkdir(parents=True, exist_ok=True)
                source = Path(meta['source_log'])
                shutil.copy2(source, folder / 'lammps.log')
                shutil.copy2(source.parent.parent / 'metadata/launch.json', folder / 'launch.json')
            log = folder / 'lammps.log'
            assert sha(log) == meta['log_sha256'], case
            launch = json.loads((folder / 'launch.json').read_text())
            assert sha(folder / 'launch.json') == meta['source_launch_sha256'], case
            text = log.read_text()
            blocks = [(h,a) for h,a in raw_blocks(log) if np.ptp(a[:,h.index('v_tool_x')]) > 1]
            h,a = blocks[0]
            segments[case] = [dict(start_step=int(b[0,h.index('Step')]), end_step=int(b[-1,h.index('Step')]),
                                   start_ps=float(b[0,h.index('Time')]), end_ps=float(b[-1,h.index('Time')])) for _,b in blocks]
            for nh, na in blocks[1:]:
                assert nh == h
                a = np.vstack((a[a[:,h.index('Time')] < na[0,h.index('Time')]],na))
            if rebuild_samples:
                from rebuild_analysis import write_csv
                t=a[:,h.index('Time')]
                theta=np.unwrap(a[:,h.index('v_phase')]*2*np.pi) if amp else np.zeros_like(t)
                z=a[:,h.index('v_tool_z')] if 'v_tool_z' in h else np.zeros_like(t)
                fields=dict(time_ps=t,travel_nm=(a[0,h.index('v_tool_x')]-a[:,h.index('v_tool_x')])/10,
                            Ft_nN=a[:,h.index('v_fx')]*1.602176634,Fz_nN=a[:,h.index('v_fz')]*1.602176634,
                            z_A=z,theta_rad=theta,vz_A_ps=2*np.pi*meta['frequency_per_ps']*amp*np.cos(theta))
                target=ROOT/'data'/f'{case}.csv'
                write_csv(target,[dict(zip(fields,values)) for values in zip(*fields.values())])
                assert sha(target)==meta['samples_sha256'],case
            samples = np.genfromtxt(ROOT/'data'/f'{case}.csv', delimiter=',', names=True)
            assert np.allclose(samples['time_ps'],a[:,h.index('Time')],atol=1e-9,rtol=0)
            assert np.allclose(samples['Ft_nN'],a[:,h.index('v_fx')]*1.602176634,atol=1e-10,rtol=0)
            t0 = float(samples['time_ps'][0])
            def variable(name):
                matches = re.findall(r'^variable '+name+r' equal ([\d.eE+-]+)\s*$',text,re.M)
                return float(matches[0]) if matches else None
            x0 = variable('tool_x0')
            z0 = variable('tool_z0')
            dt_values = re.findall(r'^timestep ([\d.eE+-]+)\s*$',text,re.M)
            assert set(dt_values)=={'0.001'}, (case,dt_values)
            dump = sorted(set(int(v) for v in re.findall(r'^dump traj all custom (\d+) ',text,re.M)))
            assert len(dump)==1
            off = meta['phase_offset']
            velocity = 2*np.pi*meta['frequency_per_ps']*amp*np.cos(2*np.pi*meta['frequency_per_ps']*samples['time_ps']+off)
            assert np.max(np.abs(velocity-samples['vz_A_ps']))<1e-5
            row=dict(case=case, condition=label, realization=rep, source=f'evidence/{case}/lammps.log',
                     translation_start_step=int(a[0,h.index('Step')]), time_origin='absolute LAMMPS Time in ps; dt=0.001 ps',
                     translation_start_ps=t0, x_start=x0, z_reference=z0,
                     phase_origin=off, phase_formula='theta=2*pi*frequency_per_ps*time_ps+phase_origin; z=z_reference+A*sin(theta)',
                     frequency_per_ps=meta['frequency_per_ps'], force_stride=int(round(np.median(np.diff(samples['time_ps']))/.001)),
                     dump_stride=dump[0], force_sample_interval_ps=float(np.median(np.diff(samples['time_ps']))),
                     trajectory_dump_interval_ps=dump[0]*.001,
                     restart_segments=len(blocks), reset_timestep_commands='; '.join(re.findall(r'^reset_timestep .*$',text,re.M)) or 'none',
                     logged_x_offset_A=float(a[0,h.index('v_tool_x')]-x0),
                     window_start_ps=t0+4/.03, window_end_ps=t0+16/.03,
                     log_sha256=sha(log), launch_sha256=sha(folder/'launch.json'))
            rows.append(row)
    with (ROOT/'data/run_provenance.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT/'data/restart_segments.json').write_text(json.dumps(segments,indent=2)+'\n')
    tex=[]
    for r in rows:
        case=r['case'].replace('scratch_base','Base').replace('_',r'\_')
        tex.append(f"{case} & {r['condition']} & {r['translation_start_ps']:.3f} & {r['force_sample_interval_ps']:.3f} & {r['trajectory_dump_interval_ps']:.3f} & {r['x_start']:.5f} & {r['restart_segments']}"+r' \\')
    (ROOT/'generated/run_manifest_rows.tex').write_text('\n'.join(tex)+'\n'+r'\bottomrule'+'\n')
    print('PASS: 12 raw logs match packaged force samples; clocks, strides, phase and segments audited')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',action='store_true');p.add_argument('--rebuild-samples',action='store_true')
    args=p.parse_args();run(args.archive,args.rebuild_samples)
