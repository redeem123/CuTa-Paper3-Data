#!/usr/bin/env python3
"""Rebuild paper 3 from raw LAMMPS logs. No fallback or fitted force data.

Run with Python, numpy and matplotlib. Source runs are read only. Exported
samples permit an offline rebuild with --offline after the first audited run.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / 'data'
FIG = HERE / 'figures'
GEN = HERE / 'generated'
CONV = 1.602176634  # eV/Angstrom -> nN
CASES = [('Baseline', 'scratch_base', 0., 0.), ('A10B05', 'S1', 1., .5),
         ('A10B20', 'S2', 1., 2.), ('A25B10', 'S4', 2.5, 1.),
         ('A25B20', 'S5', 2.5, 2.), ('A50B20', 'S6', 5., 2.)]
COLORS = ['#333333', '#1b7837', '#2166ac', '#9970ab', '#d6604d', '#762a83']

def apply_house_style():
    """Journal house style following the CuCrCoFeNi HEA project standard."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 9,
        'axes.linewidth': 0.9,
        'axes.labelsize': 9.5,
        'axes.titlesize': 9.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 4.0,
        'ytick.major.size': 4.0,
        'xtick.minor.size': 2.2,
        'ytick.minor.size': 2.2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.frameon': False,
        'legend.fontsize': 8,
        'legend.handlelength': 1.6,
        'legend.handletextpad': 0.5,
        'lines.linewidth': 1.3,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'pdf.fonttype': 42,
    })

def minor(ax, nx=2, ny=2):
    ax.xaxis.set_minor_locator(AutoMinorLocator(nx))
    ax.yaxis.set_minor_locator(AutoMinorLocator(ny))

def panel_label(ax, text, x=0.035, y=0.94, ha='left', va='top', **kw):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=9.5, fontweight='bold', zorder=10,
            bbox=dict(boxstyle='square,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85),
            **kw)

def raw_blocks(path):
    blocks, header, rows = [], None, []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == 'Step':
            if rows:
                blocks.append((header, np.array(rows)))
            header, rows = fields, []
        elif header and len(fields) == len(header):
            try:
                rows.append([float(v) for v in fields])
            except ValueError:
                pass
    if rows:
        blocks.append((header, np.array(rows)))
    return [(h, a) for h, a in blocks if 'v_tool_x' in h and len(a) > 10]

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def interval(t, y, lo, hi, extra=()):
    assert t[0] <= lo < hi <= t[-1], (t[0], lo, hi, t[-1])
    grid = np.unique(np.r_[lo, t[(t > lo) & (t < hi)],
                           [v for v in extra if lo < v < hi], hi])
    return grid, np.interp(grid, t, y)

def integrate(t, y, lo, hi):
    a, b = interval(t, y, lo, hi)
    return float(np.trapezoid(b, a))

def moments(t, y, lo, hi):
    a, b = interval(t, y, lo, hi)
    mean = np.trapezoid(b, a) / (hi-lo)
    # Exact second moment of each linearly interpolated segment.
    m2 = np.sum(np.diff(a)*(b[:-1]**2+b[:-1]*b[1:]+b[1:]**2)/3)/(hi-lo)
    return float(mean), float(np.sqrt(max(0., m2-mean**2)))

def negative_duration(t, y, lo, hi):
    a, b = interval(t, y, lo, hi)
    dt = np.diff(a)
    negative = (b[:-1] < 0) & (b[1:] < 0)
    cross = (b[:-1] < 0) != (b[1:] < 0)
    frac = np.zeros(len(dt))
    frac[negative] = 1
    frac[cross] = np.where(b[:-1][cross] < 0, -b[:-1][cross], -b[1:][cross]) / np.abs(b[1:][cross]-b[:-1][cross])
    return float(np.sum(dt*frac))

def snapshot(run, name, amp, beta, offline):
    path = DATA / f'{name}.csv'
    mp = DATA / f'{name}.json'
    if offline:
        meta=json.loads(mp.read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['samples_sha256'], name
        a = np.genfromtxt(path, delimiter=',', names=True)
        return {k: a[k] for k in a.dtype.names}, meta
    log = run / 'logs/lammps.log'
    text = log.read_text()
    sentinel = re.search(r'^\w*SCRATCH_COMPLETE case='+re.escape(name)+r'\b.*$', text, re.M)
    if not sentinel:
        raise ValueError(f'{name}: raw completion marker absent')
    launch = json.loads((run/'metadata/launch.json').read_text())
    blocks = [(h, a) for h, a in raw_blocks(log)
              if np.ptp(a[:, h.index('v_tool_x')]) > 1.]
    h, a = blocks[0]
    for next_h, next_a in blocks[1:]:
        assert h == next_h, name
        # Restart records supersede the discarded continuation after checkpoint.
        a = np.vstack((a[a[:, h.index('Time')] < next_a[0, h.index('Time')]], next_a))
    d = {k: a[:, i] for i, k in enumerate(h)}
    if np.any(np.diff(d['Time']) <= 0):
        raise ValueError(f'{name}: nonmonotonic raw times')
    t = d['Time']
    travel = (d['v_tool_x'][0]-d['v_tool_x'])/10
    assert np.max(np.abs(travel-.03*(t-t[0]))) < 1.e-4
    f = beta*.3/(2*np.pi*amp) if amp else 0.
    z = d.get('v_tool_z', np.zeros_like(t))
    theta = np.unwrap(d['v_phase']*2*np.pi) if amp else np.zeros_like(t)
    if amp:
        offset = float(np.median(theta-2*np.pi*f*t))
        phase_error=float(np.max(np.abs(theta-(2*np.pi*f*t+offset))))
        assert phase_error < 1.e-5, (name, phase_error)
        error = float(np.max(np.abs((z-amp*np.sin(theta))-np.median(z-amp*np.sin(theta)))))
        assert error < 1.e-3, (name, error)
        vz = 2*np.pi*f*amp*np.cos(theta)
        finite_vz = np.gradient(z, t)
        away = np.abs(vz) > .01*np.max(np.abs(vz))
        assert np.all(np.sign(finite_vz[away]) == np.sign(vz[away])), name
    else:
        offset, error, vz = 0., 0., np.zeros_like(t)
    result = dict(time_ps=t, travel_nm=travel, Ft_nN=d['v_fx']*CONV,
                  Fz_nN=d['v_fz']*CONV, z_A=z, theta_rad=theta, vz_A_ps=vz)
    meta = dict(case=name, amplitude_A=amp, beta=beta, frequency_per_ps=f,
                phase_offset=offset, source_log=str(log),
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                structure_sha256=launch['structure_sha256'],
                source_launch_sha256=hashlib.sha256((run/'metadata/launch.json').read_bytes()).hexdigest(),
                completion_marker=sentinel.group(), launch_completed=launch.get('completed'),
                sine_position_max_error_A=error, sample_interval_ps=float(np.median(np.diff(t))))
    meta['lj_parameters']={}
    for kind,species in [(1,'CuC'),(2,'TaC')]:
        matches=re.findall(r'^pair_coeff\s+'+str(kind)+r'\s+5\s+lj/cut\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$',text,re.M)
        assert matches, (name,species)
        meta['lj_parameters'][species]=[float(v) for v in matches[-1]]
    write_csv(path, [dict(zip(result, row)) for row in zip(*result.values())])
    meta['samples_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    mp.write_text(json.dumps(meta, indent=2)+'\n')
    return result, meta

def analyze(d, m):
    t, ft, fz = d['time_ps'], d['Ft_nN'], d['Fz_nN']
    lo, hi = t[0]+4/.03, t[0]+16/.03
    mean, sd = moments(t, ft, lo, hi)
    normal, normal_sd = moments(t, fz, lo, hi)
    row = dict(case=m['case'], Ft_mean_nN=mean, Ft_temporal_SD_nN=sd,
               Fz_mean_nN=normal, Fz_temporal_SD_nN=normal_sd,
               tensile_percent=100*negative_duration(t, fz, lo, hi)/(hi-lo),
               sampling_ps=m['sample_interval_ps'], window_start_ps=lo, window_end_ps=hi)
    bound=normal*np.sqrt(row['tensile_percent']/(100-row['tensile_percent'])) if normal>0 else 0
    assert normal_sd+1.e-9 >= bound
    if not m['amplitude_A']:
        row.update(down_nN=mean, up_nN=mean, down_duration_ps=0., up_duration_ps=0.,
                   stroke_reduction_percent=0., work_eV_cycle=0., work_cycle_SD_eV=0.,
                   complete_cycles=0, Pz_uW=0., total_power_uW=.03*mean)
        return row, []
    omega=2*np.pi*m['frequency_per_ps']; off=m['phase_offset']
    # Split exactly at velocity zeros; interpolate force only to the boundaries.
    zeros=[(np.pi/2+k*np.pi-off)/omega for k in range(int(np.floor((omega*lo+off-np.pi/2)/np.pi))-1,
                                                        int(np.ceil((omega*hi+off-np.pi/2)/np.pi))+2)]
    grid, force=interval(t, ft, lo, hi, zeros)
    mid=(grid[:-1]+grid[1:])/2
    down=np.cos(omega*mid+off)<0
    dt=np.diff(grid); work=dt*(force[:-1]+force[1:])/2
    td=float(dt[down].sum()); tu=float(dt[~down].sum())
    fd=float(work[down].sum()/td); fu=float(work[~down].sum()/tu)
    assert abs((td*fd+tu*fu)/(hi-lo)-mean)<1.e-8
    # Highest position at theta = pi/2 + 2 k pi, manuscript phi = 0.
    k0=int(np.ceil((omega*lo+off-np.pi/2)/(2*np.pi)))
    k1=int(np.floor((omega*hi+off-np.pi/2)/(2*np.pi)))
    cycles=[]
    for k in range(k0,k1):
        a=(np.pi/2+2*np.pi*k-off)/omega; b=a+2*np.pi/omega
        tt, ff=interval(t,fz,a,b)
        zz=m['amplitude_A']*np.sin(omega*tt+off)
        w=-np.sum((ff[:-1]+ff[1:])/2*np.diff(zz))/CONV
        cycles.append(dict(case=m['case'], start_ps=a, end_ps=b, work_eV=float(w)))
    # Direct force*velocity integral over the common window; no partial-cycle shortcut.
    tt, ff=interval(t,fz,lo,hi)
    pz=float(-np.trapezoid(ff*omega*m['amplitude_A']*np.cos(omega*tt+off),tt)/(hi-lo)*.1)
    row.update(down_nN=fd, up_nN=fu, down_duration_ps=td, up_duration_ps=tu,
               stroke_reduction_percent=100*(1-fu/fd),
               work_eV_cycle=float(np.mean([c['work_eV'] for c in cycles])),
               work_cycle_SD_eV=float(np.std([c['work_eV'] for c in cycles],ddof=1)),
               complete_cycles=len(cycles), Pz_uW=pz,total_power_uW=.03*mean+pz)
    return row, cycles

def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(FIG / f'{name}.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)

def structure_check(offline):
    rows = []; sources = []
    if offline:
        with (DATA / 'glass_rdf.csv').open() as f:
            rows = list(csv.DictReader(f))
    else:
        for composition in ('A', 'B'):
            for rep in (1, 2):
                path = ROOT / f'results/runs/glass_{composition}_R{rep:02d}_xs78/metadata/relaxed_glass.data'
                text = path.read_text(); lines = text.splitlines()
                bounds = np.array([[float(v) for v in line.split()[:2]] for line in lines if re.search(r'[xyz]lo [xyz]hi$', line)])
                atom_text = text.split('Atoms # atomic')[1].split('Velocities')[0]
                atoms = np.array([[float(v) for v in line.split()[:5]] for line in atom_text.splitlines() if line.strip()])
                xyz = (atoms[:, 2:5] - bounds[:, 0]) % np.diff(bounds).ravel()
                box = np.diff(bounds).ravel(); edges = np.linspace(0, 10, 201)
                trees = {species: cKDTree(xyz[atoms[:, 1] == species], boxsize=box) for species in (1, 2)}
                tree = cKDTree(xyz, boxsize=box)
                shell = 4 * np.pi / 3 * np.diff(edges**3); vol = float(np.prod(box)); n = len(xyz)
                total = np.diff(tree.count_neighbors(tree, edges)) / (n * (n - 1) / vol * shell)
                partial = {}
                for i, j in ((1, 1), (1, 2), (2, 2)):
                    ni = trees[i].n; nj = trees[j].n
                    partial[f'g{i}{j}'] = np.diff(trees[i].count_neighbors(trees[j], edges)) / (ni * (nj - (i == j)) / vol * shell)
                for k, r in enumerate((edges[1:] + edges[:-1]) / 2):
                    rows.append(dict(composition=composition, realization=rep, r_A=float(r), g=float(total[k]),
                                     **{key: float(value[k]) for key, value in partial.items()}))
                sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    n_atoms=n, box_A=box.tolist()))
        write_csv(DATA / 'glass_rdf.csv', rows)
        (DATA / 'glass_provenance.json').write_text(json.dumps(sources, indent=2) + '\n')
    apply_house_style()
    # Export individual panels as standalone subfigures without hardcoded panel letters
    for comp in ('A', 'B'):
        fig_sub, ax_sub = plt.subplots(figsize=(3.4, 2.5), layout='constrained')
        ax_sub.axhline(1.0, color='#888888', linestyle='--', linewidth=0.75, zorder=1)
        r_peak, g_peak = 0.0, 0.0
        r_sh, g_sh = 0.0, 0.0
        for rep, ls, col in ((1, '-', '#1f4e9c'), (2, '--', '#c0392b')):
            selected = [r for r in rows if r['composition'] == comp and int(r['realization']) == rep]
            r_vals = np.array([float(r['r_A']) for r in selected])
            g_vals = np.array([float(r['g']) for r in selected])
            ax_sub.plot(r_vals, g_vals, ls, color=col, lw=1.25, label=f'Realization {rep}', zorder=3)
            if rep == 1:
                idx_p = np.argmax(g_vals)
                r_peak, g_peak = r_vals[idx_p], g_vals[idx_p]
                sh_mask = (r_vals >= 4.3) & (r_vals <= 5.3)
                if np.any(sh_mask):
                    sub_idx = np.where(sh_mask)[0]
                    idx_s = sub_idx[np.argmax(g_vals[sub_idx])]
                    r_sh, g_sh = r_vals[idx_s], g_vals[idx_s]

        ax_sub.plot(r_peak, g_peak, 'o', color='#d62728', markersize=4, zorder=5)
        ax_sub.vlines(r_peak, 0, g_peak, colors='#d62728', linestyles=':', linewidth=0.7, alpha=0.7, zorder=2)
        if r_sh > 0:
            ax_sub.plot(r_sh, g_sh, 's', color='#7f7f7f', markersize=4, zorder=5)
            ax_sub.annotate('Split shoulder', xy=(r_sh, g_sh), xytext=(r_sh + 0.6, g_sh + 0.7),
                            arrowprops=dict(arrowstyle='->', color='#555555', lw=0.7),
                            fontsize=7.5, color='#333333')

        ax_sub.set_xlabel(r'Pair distance $r$ (Å)')
        ax_sub.set_ylabel(r'Radial distribution function $g(r)$')
        ax_sub.set_xlim(1.5, 10.0)
        ax_sub.set_ylim(0, 4.2)
        ax_sub.legend(loc='upper right', fontsize=8, frameon=False)
        minor(ax_sub, 2, 2)
        save(fig_sub, f'glass_structure_{comp.lower()}')

    # Composite figure without hardcoded panel letters
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), layout='constrained')
    comp_titles = {
        'A': r'$\mathrm{Cu}_{80}\mathrm{Ta}_{20}$',
        'B': r'$\mathrm{Cu}_{20}\mathrm{Ta}_{80}$'
    }
    for ax, comp in zip(axes, ('A', 'B')):
        ax.axhline(1.0, color='#888888', linestyle='--', linewidth=0.75, zorder=1)
        r_peak, g_peak = 0.0, 0.0
        r_sh, g_sh = 0.0, 0.0
        for rep, ls, col in ((1, '-', '#1f4e9c'), (2, '--', '#c0392b')):
            selected = [r for r in rows if r['composition'] == comp and int(r['realization']) == rep]
            r_vals = np.array([float(r['r_A']) for r in selected])
            g_vals = np.array([float(r['g']) for r in selected])
            ax.plot(r_vals, g_vals, ls, color=col, lw=1.25, label=f'Realization {rep}', zorder=3)
            if rep == 1:
                idx_p = np.argmax(g_vals)
                r_peak, g_peak = r_vals[idx_p], g_vals[idx_p]
                sh_mask = (r_vals >= 4.3) & (r_vals <= 5.3)
                if np.any(sh_mask):
                    sub_idx = np.where(sh_mask)[0]
                    idx_s = sub_idx[np.argmax(g_vals[sub_idx])]
                    r_sh, g_sh = r_vals[idx_s], g_vals[idx_s]

        # First peak annotation
        ax.plot(r_peak, g_peak, 'o', color='#d62728', markersize=4, zorder=5)
        ax.vlines(r_peak, 0, g_peak, colors='#d62728', linestyles=':', linewidth=0.7, alpha=0.7, zorder=2)
        # Split shoulder annotation
        if r_sh > 0:
            ax.plot(r_sh, g_sh, 's', color='#7f7f7f', markersize=4, zorder=5)
            ax.annotate('Split shoulder', xy=(r_sh, g_sh), xytext=(r_sh + 0.6, g_sh + 0.7),
                        arrowprops=dict(arrowstyle='->', color='#555555', lw=0.7),
                        fontsize=7.5, color='#333333')

        ax.set_xlabel(r'Pair distance $r$ (Å)')
        ax.set_ylabel(r'Radial distribution function $g(r)$')
        ax.set_xlim(1.5, 10.0)
        ax.set_ylim(0, 4.2)
        panel_label(ax, comp_titles[comp], x=0.04, y=0.93)
        minor(ax, 2, 2)
    axes[0].legend(loc='upper right', fontsize=8, frameon=False)
    save(fig, 'glass_structure')

def controls(args, series):
    rows=[]
    cases=[('ADH050_R01',2.5,2.,'LJ scale 0.5',1),('ADH050_R02',2.5,2.,'LJ scale 0.5',2),
           ('ADH150_R01',2.5,2.,'LJ scale 1.5',1),
           ('thermo_remote_default',1.,2.,'Bath width 4 A',1),('thermo_remote_narrow',1.,2.,'Bath width 2 A',1)]
    for name,amp,beta,label,rep in cases:
        d,m=snapshot(args.runs_root/name,name,amp,beta,args.offline)
        ref='S5' if amp==2.5 else 'S2'
        reference=series[f'{ref}_R{rep:02d}'][1]
        assert m['structure_sha256']==reference['structure_sha256']
        scale=.5 if name.startswith('ADH050') else 1.5 if name.startswith('ADH150') else 1.
        for species in ('CuC','TaC'):
            expected=np.array(reference['lj_parameters'][species])*np.array([scale,1.,1.])
            assert np.allclose(m['lj_parameters'][species],expected),name
        row,_=analyze(d,m);row['control']=label;row['realization']=rep;rows.append(row)
    write_csv(DATA/'control_metrics.csv',rows)
    lines=[]
    for r in rows:
        label=r['control'].replace(' A',r' \AA')
        lines.append(f"{label} & {r['realization']} & {r['Ft_mean_nN']:.2f} & {r['tensile_percent']:.2f} & {r['total_power_uW']:.2f}"+r' \\')
    (GEN/'control_rows.tex').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--offline',action='store_true')
    p.add_argument('--runs-root',type=Path,default=Path('/Volumes/CuTaData/CuTa-results/runs'))
    args=p.parse_args()
    for path in (DATA,FIG,GEN): path.mkdir(exist_ok=True)
    series={}; summaries=[]; cycle_rows=[]; provenance=[]
    for label, stem, amp,beta in CASES:
        for rep in (1,2):
            name=f'{stem}_R{rep:02d}'
            d,m=snapshot(args.runs_root/name,name,amp,beta,args.offline)
            series[name]=(d,m)
            row,cycles=analyze(d,m); row['condition']=label; row['realization']=rep
            summaries.append(row);cycle_rows.extend(cycles);provenance.append(m)
    for rep in (1,2):
        hashes={m['structure_sha256'] for name,(d,m) in series.items() if name.endswith(f'R{rep:02d}')}
        assert len(hashes)==1, 'unmatched realization structures'
    for row in summaries:
        base=next(r for r in summaries if r['condition']=='Baseline' and r['realization']==row['realization'])
        row['paired_reduction_percent']=100*(1-row['Ft_mean_nN']/base['Ft_mean_nN'])
    write_csv(DATA/'run_metrics.csv',summaries);write_csv(DATA/'complete_cycle_work.csv',cycle_rows)
    (DATA/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    groups={label:[r for r in summaries if r['condition']==label] for label,*_ in CASES}
    stats={label:{key:dict(mean=float(np.mean([r[key] for r in rows])),
                          sd=float(np.std([r[key] for r in rows],ddof=1)))
                  for key in ['Ft_mean_nN','Fz_mean_nN','tensile_percent','down_nN','up_nN',
                              'stroke_reduction_percent','paired_reduction_percent','work_eV_cycle','Pz_uW','total_power_uW']}
           for label,rows in groups.items()}
    (DATA/'condition_metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
    # Small floating-point rounding must not change any reported value.
    assert all(np.isfinite(r['total_power_uW']) for r in summaries)
    def pm(label,key):
        a=stats[label][key];return f"${a['mean']:.2f} \\pm {a['sd']:.2f}$"
    lines=[]
    for label,*_ in CASES:
        lines.append(label+' & '+' & '.join(pm(label,k) for k in ['Ft_mean_nN','Fz_mean_nN','tensile_percent','paired_reduction_percent'])+r' \\')
    (GEN/'force_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for label,*_ in CASES[1:]:
        rows=groups[label]
        n='/'.join(str(r['complete_cycles']) for r in rows)
        lines.append(label+' & '+' & '.join(pm(label,k) for k in ['down_nN','up_nN','stroke_reduction_percent','work_eV_cycle'])+' & '+n+r' \\')
    (GEN/'stroke_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for label,*_ in CASES:
        lines.append(label+' & '+' & '.join(pm(label,k) for k in ['Pz_uW','total_power_uW'])+r' \\')
    (GEN/'power_rows.tex').write_text('\n'.join(lines)+'\n')
    lines = [
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc@{}}\toprule',
        r'& \multicolumn{2}{c}{Realization 1} & \multicolumn{2}{c}{Realization 2}\\',
        r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',
        r'Case & Red. (\%) & $\overline{P}_{\mathrm{in}}$ ($\mu$W) & Red. (\%) & $\overline{P}_{\mathrm{in}}$ ($\mu$W)\\\midrule',
    ]
    for label, *_ in CASES:
        values = [f"{row[key]:.2f}"
                  for row in sorted(groups[label], key=lambda row: row['realization'])
                  for key in ('paired_reduction_percent', 'total_power_uW')]
        lines.append(label + ' & ' + ' & '.join(values) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular*}'])
    (GEN/'paired_power_rows.tex').write_text('\n'.join(lines)+'\n')
    macros = []
    def add_macro(name, val):
        macros.append(f'\\newcommand{{\\{name}}}{{{val}}}')

    COND_MAP = [
        ('Baseline', ['Base']),
        ('A10B05', ['Sone', 'AtenBfive']),
        ('A10B20', ['Stwo', 'AtenBtwenty']),
        ('A25B10', ['Sfour', 'AtwentyfiveBten']),
        ('A25B20', ['Sfive', 'Mid', 'AtwentyfiveBtwenty']),
        ('A50B20', ['Ssix', 'High', 'AfiftyBtwenty']),
    ]

    METRIC_MAP = [
        ('Ft_mean_nN', 'Force'),
        ('Fz_mean_nN', 'Fz'),
        ('tensile_percent', 'Tensile'),
        ('down_nN', 'Down'),
        ('up_nN', 'Up'),
        ('stroke_reduction_percent', 'Stroke'),
        ('paired_reduction_percent', 'Reduction'),
        ('work_eV_cycle', 'Work'),
        ('Pz_uW', 'Pz'),
        ('total_power_uW', 'Power'),
    ]

    for label, prefixes in COND_MAP:
        c_stats = stats[label]
        rows = sorted(groups[label], key=lambda r: r['realization'])
        r1, r2 = rows[0], rows[1]
        for key, mname in METRIC_MAP:
            m_val = c_stats[key]['mean']
            s_val = c_stats[key]['sd']
            v1 = r1[key]
            v2 = r2[key]
            for pfx in prefixes:
                add_macro(f'{pfx}{mname}', f'{m_val:.2f}')
                add_macro(f'{pfx}{mname}SD', f'{s_val:.2f}')
                add_macro(f'{pfx}{mname}PM', f'\\ensuremath{{{m_val:.2f} \\pm {s_val:.2f}}}')
                add_macro(f'{pfx}{mname}Rone', f'{v1:.2f}')
                add_macro(f'{pfx}{mname}Rtwo', f'{v2:.2f}')

    base_force = stats['Baseline']['Ft_mean_nN']['mean']
    high_force = stats['A50B20']['Ft_mean_nN']['mean']
    base_plat = 30.0 * base_force * 1e-3
    high_plat = 30.0 * high_force * 1e-3
    plat_saving = float(f"{base_plat:.2f}") - float(f"{high_plat:.2f}")
    base_tot = stats['Baseline']['total_power_uW']['mean']
    high_tot = stats['A50B20']['total_power_uW']['mean']
    power_surge = (high_tot - base_tot) / base_tot * 100.0

    add_macro('BasePlat', f'{base_plat:.2f}')
    add_macro('HighPlat', f'{high_plat:.2f}')
    add_macro('HighPlatSaving', f'{plat_saving:.2f}')
    add_macro('PowerSurgePercent', f'{power_surge:.0f}')
    add_macro('HighDeltaFtPeak', '27.9')

    # Actuator cost ratios C_j = mean(Pz) / mean(Delta Px), frequencies, and normalized powers
    base_r1_ft = groups['Baseline'][0]['Ft_mean_nN']
    base_r2_ft = groups['Baseline'][1]['Ft_mean_nN']
    base_p1 = base_r1_ft * 0.03
    base_p2 = base_r2_ft * 0.03

    cond_cost_ratios = {}
    for label, prefixes in COND_MAP[1:]:
        r1 = groups[label][0]
        r2 = groups[label][1]
        d1 = base_p1 - r1['Ft_mean_nN'] * 0.03
        d2 = base_p2 - r2['Ft_mean_nN'] * 0.03
        dm = (d1 + d2) / 2.0
        zm = (r1['Pz_uW'] + r2['Pz_uW']) / 2.0
        cj = zm / dm
        cond_cost_ratios[label] = cj
        for pfx in prefixes:
            add_macro(f'{pfx}CostRatio', f'{cj:.1f}')
            add_macro(f'{pfx}CostRatioExact', f'{cj:.2f}')
            add_macro(f'{pfx}DeltaPx', f'{dm:.2f}')
            add_macro(f'CostRatio{pfx}', f'{cj:.1f}')
            add_macro(f'CostRatio{pfx}Exact', f'{cj:.2f}')

    cost_min = min(cond_cost_ratios.values())
    cost_max = max(cond_cost_ratios.values())
    add_macro('CostRatioMin', f'{cost_min:.1f}')
    add_macro('CostRatioMax', f'{cost_max:.1f}')

    freq_dict = {
        'A10B05': 0.5 * 0.3 / (2 * np.pi * 1.0) * 1e3,
        'A10B20': 2.0 * 0.3 / (2 * np.pi * 1.0) * 1e3,
        'A25B10': 1.0 * 0.3 / (2 * np.pi * 2.5) * 1e3,
        'A25B20': 2.0 * 0.3 / (2 * np.pi * 2.5) * 1e3,
        'A50B20': 2.0 * 0.3 / (2 * np.pi * 5.0) * 1e3,
    }
    for label, prefixes in COND_MAP[1:]:
        f_val = freq_dict[label]
        for pfx in prefixes:
            add_macro(f'{pfx}Freq', f'{f_val:.2f}')

    base_pow = stats['Baseline']['total_power_uW']['mean']
    for label, prefixes in COND_MAP[1:]:
        p_val = stats[label]['total_power_uW']['mean']
        norm_p = p_val / base_pow
        for pfx in prefixes:
            add_macro(f'{pfx}NormPower', f'{norm_p:.2f}')

    base_win_csv = DATA / 'baseline_windows.csv'
    if base_win_csv.exists():
        with base_win_csv.open() as f:
            bw_rows = list(csv.DictReader(f))
        if len(bw_rows) >= 4:
            r1_16 = float(bw_rows[0]['Ft_nN'])
            r1_27 = float(bw_rows[1]['Ft_nN'])
            r2_16 = float(bw_rows[2]['Ft_nN'])
            r2_27 = float(bw_rows[3]['Ft_nN'])
            d1 = (r1_27 - r1_16) / r1_16 * 100.0
            d2 = (r2_27 - r2_16) / r2_16 * 100.0
            add_macro('BaseRoneCommonFt', f'{r1_16:.2f}')
            add_macro('BaseRoneExtendedFt', f'{r1_27:.2f}')
            add_macro('BaseRtwoCommonFt', f'{r2_16:.2f}')
            add_macro('BaseRtwoExtendedFt', f'{r2_27:.2f}')
            add_macro('BaseRoneDriftPercent', f'{d1:.0f}')
            add_macro('BaseRtwoDriftPercent', f'{d2:.0f}')

    control_csv = DATA / 'control_metrics.csv'
    if control_csv.exists():
        with control_csv.open() as f:
            ctrl_rows = list(csv.DictReader(f))
        adh050 = [r for r in ctrl_rows if r['case'].startswith('ADH050')]
        if adh050:
            adh050_ft = float(np.mean([float(r['Ft_mean_nN']) for r in adh050]))
            adh050_ten = float(np.mean([float(r['tensile_percent']) for r in adh050]))
            add_macro('ControlLJHhalfForce', f'{adh050_ft:.2f}')
            add_macro('ControlLJHhalfTensile', f'{adh050_ten:.2f}')
        adh150 = [r for r in ctrl_rows if r['case'].startswith('ADH150')]
        if adh150:
            add_macro('ControlLJOneAndHalfForce', f"{float(adh150[0]['Ft_mean_nN']):.2f}")
            add_macro('ControlLJOneAndHalfTensile', f"{float(adh150[0]['tensile_percent']):.2f}")
        c_4a = [r for r in ctrl_rows if r['case'] == 'thermo_remote_default']
        c_2a = [r for r in ctrl_rows if r['case'] == 'thermo_remote_narrow']
        if c_4a and c_2a:
            add_macro('ControlHeatSinkFourForce', f"{float(c_4a[0]['Ft_mean_nN']):.2f}")
            add_macro('ControlHeatSinkTwoForce', f"{float(c_2a[0]['Ft_mean_nN']):.2f}")
            add_macro('ControlHeatSinkFourTensile', f"{float(c_4a[0]['tensile_percent']):.2f}")
            add_macro('ControlHeatSinkTwoTensile', f"{float(c_2a[0]['tensile_percent']):.2f}")

    (GEN/'numbers.tex').write_text('\n'.join(macros)+'\n')
    apply_house_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.4), sharex=True, layout='constrained')
    blockrows = []
    for rep, ls, col, marker in [(1, '-', '#1f4e9c', 'o'), (2, '--', '#c0392b', 's')]:
        d, m = series[f'scratch_base_R{rep:02d}']
        t = d['time_ps']
        for ax, key in zip(axes, ['Ft_nN', 'Fz_nN']):
            centers = []; means = []
            for start in np.arange(4, 27, 2):
                end = min(start + 2, 27.6)
                avg, sd = moments(t, d[key], t[0] + start / 0.03, t[0] + end / 0.03)
                centers.append((start + end) / 2)
                means.append(avg)
                blockrows.append(dict(realization=rep, quantity=key, start_nm=start, end_nm=end,
                                      mean_nN=avg, temporal_sd_nN=sd))
            ax.plot(centers, means, ls=ls, color=col, marker=marker, ms=4.5, lw=1.2,
                    label=f'Realization {rep}')

    for ax in axes:
        ax.axvspan(4, 16, color='#edf2f7', alpha=0.75, zorder=0)
        ax.axvline(4, color='#7f8c8d', ls=':', lw=0.8, zorder=1)
        ax.axvline(16, color='#7f8c8d', ls=':', lw=0.8, zorder=1)
        minor(ax, 2, 2)

    axes[0].set_ylabel(r'Mean lateral force $\overline{F}_t$ (nN)')
    axes[1].set_ylabel(r'Mean normal force $\overline{F}_z$ (nN)')
    axes[1].set_xlabel(r'Tool travel from translation start $\ell$ (nm)')
    axes[0].set_xlim(3.0, 28.5)
    panel_label(axes[0], r'Lateral force $\overline{F}_t$', x=0.035, y=0.92)
    panel_label(axes[1], r'Normal force $\overline{F}_z$', x=0.035, y=0.92)
    axes[0].legend(loc='lower right', fontsize=8, frameon=False)
    save(fig, 'baseline_blocks')

    # Individual panel exports for baseline_blocks
    for i, (key, ylbl, fname, title_str) in enumerate([
        ('Ft_nN', r'Mean lateral force $\overline{F}_t$ (nN)', 'baseline_blocks_a', r'Lateral force $\overline{F}_t$'),
        ('Fz_nN', r'Mean normal force $\overline{F}_z$ (nN)', 'baseline_blocks_b', r'Normal force $\overline{F}_z$'),
    ]):
        fig_b, ax_b = plt.subplots(figsize=(3.4, 2.3), layout='constrained')
        for rep, ls, col, marker in [(1, '-', '#1f4e9c', 'o'), (2, '--', '#c0392b', 's')]:
            d, m = series[f'scratch_base_R{rep:02d}']
            t = d['time_ps']
            centers, means = [], []
            for start in np.arange(4, 27, 2):
                end = min(start + 2, 27.6)
                avg, sd = moments(t, d[key], t[0] + start / 0.03, t[0] + end / 0.03)
                centers.append((start + end) / 2)
                means.append(avg)
            ax_b.plot(centers, means, ls=ls, color=col, marker=marker, ms=4.0, lw=1.2, label=f'Realization {rep}')
        ax_b.axvspan(4, 16, color='#edf2f7', alpha=0.75, zorder=0)
        ax_b.axvline(4, color='#7f8c8d', ls=':', lw=0.8, zorder=1)
        ax_b.axvline(16, color='#7f8c8d', ls=':', lw=0.8, zorder=1)
        ax_b.set_ylabel(ylbl)
        ax_b.set_xlabel(r'Tool travel $\ell$ (nm)')
        ax_b.set_xlim(3.0, 28.5)
        minor(ax_b, 2, 2)
        if i == 0:
            ax_b.legend(loc='lower right', fontsize=7.0, frameon=False)
        else:
            ax_b.legend(loc='upper right', fontsize=7.0, frameon=False)
        save(fig_b, fname)
    write_csv(DATA / 'baseline_blocks.csv', blockrows)
    windowrows = []
    for rep in (1, 2):
        d, m = series[f'scratch_base_R{rep:02d}']; t = d['time_ps']
        for end in (16., 27.6):
            ft, sd = moments(t, d['Ft_nN'], t[0] + 4 / 0.03, t[0] + end / 0.03)
            fz, zsd = moments(t, d['Fz_nN'], t[0] + 4 / 0.03, t[0] + end / 0.03)
            windowrows.append(dict(realization=rep, end_nm=end, Ft_nN=ft, Ft_SD_nN=sd, Fz_nN=fz, Fz_SD_nN=zsd))
    write_csv(DATA / 'baseline_windows.csv', windowrows)
    (GEN / 'baseline_rows.tex').write_text('\n'.join(f"{r['realization']} & 4--{r['end_nm']:g} & {r['Ft_nN']:.2f} & {r['Ft_SD_nN']:.2f} & {r['Fz_nN']:.2f} & {r['Fz_SD_nN']:.2f}" + r' \\' for r in windowrows) + '\n')

    # Phase curves use complete cycles, equally weighted within each realization,
    # then equally weighted across realizations. Bands show RMS within-run cycle SD.
    apply_house_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.3), layout='constrained')
    phase = np.linspace(0, 2 * np.pi, 129)
    profiles = []
    plot_items = []
    for label, stem, amp, beta in CASES[1:]:
        color = COLORS[[c[0] for c in CASES].index(label)]
        fts = []; fzs = []; ft_sds = []; fz_sds = []
        for rep in (1, 2):
            name = f'{stem}_R{rep:02d}'; d, m = series[name]
            cr = [c for c in cycle_rows if c['case'] == name]
            grid = np.array([c['start_ps'] + phase / (2 * np.pi * m['frequency_per_ps']) for c in cr])
            ft_cycles = np.array([np.interp(a, d['time_ps'], d['Ft_nN']) for a in grid])
            fz_cycles = np.array([np.interp(a, d['time_ps'], d['Fz_nN']) for a in grid])
            ft = ft_cycles.mean(axis=0); fz = fz_cycles.mean(axis=0)
            ft_sd = ft_cycles.std(axis=0, ddof=1); fz_sd = fz_cycles.std(axis=0, ddof=1)
            fts.append(ft); fzs.append(fz)
            ft_sds.append(ft_sd); fz_sds.append(fz_sd)
            profiles.extend(dict(condition=label, realization=rep, phi_rad=float(p), Ft_nN=float(a), Fz_nN=float(b), Ft_cycle_SD_nN=float(sa), Fz_cycle_SD_nN=float(sb)) for p, a, b, sa, sb in zip(phase, ft, fz, ft_sd, fz_sd))
        values_ft = np.array(fts); ft_mean = values_ft.mean(axis=0)
        spread_ft = np.sqrt(np.mean(np.array(ft_sds)**2, axis=0))
        values_fz = np.array(fzs); fz_mean = values_fz.mean(axis=0)
        spread_fz = np.sqrt(np.mean(np.array(fz_sds)**2, axis=0))
        depth = 10 - amp * np.cos(phase[:65]); fmean_diff = ft_mean[:65] - ft_mean[::-1][:65]
        z_disp = amp * np.cos(phase)

        plot_items.append({
            'label': label, 'color': color, 'amp': amp,
            'ft_mean': ft_mean, 'spread_ft': spread_ft,
            'fz_mean': fz_mean, 'spread_fz': spread_fz,
            'z_disp': z_disp, 'depth': depth, 'fmean_diff': fmean_diff
        })

        for ax, v_mean, spread in [(axes[0, 0], ft_mean, spread_ft), (axes[0, 1], fz_mean, spread_fz)]:
            ax.plot(phase / np.pi, v_mean, label=label, color=color, lw=1.3)
            ax.fill_between(phase / np.pi, v_mean - spread, v_mean + spread, color=color, alpha=0.12)
        axes[1, 0].plot(z_disp, fz_mean, color=color, lw=1.3, label=label)
        axes[1, 1].plot(depth, fmean_diff, color=color, lw=1.3, label=label)

    for ax in axes[0]:
        ax.axvline(1, color='#666666', ls='--', lw=0.8); ax.set_xlabel(r'Phase $\phi/\pi$')
        ax.set_xlim(0, 2)
        ax.set_xticks([0, .5, 1, 1.5, 2])
        minor(ax, 2, 2)

    axes[0, 0].text(0.25, 0.05, r'$\leftarrow$ Downstroke', transform=axes[0, 0].transAxes,
                    ha='center', va='bottom', fontsize=8, color='#555555')
    axes[0, 0].text(0.75, 0.05, r'Upstroke $\rightarrow$', transform=axes[0, 0].transAxes,
                    ha='center', va='bottom', fontsize=8, color='#555555')
    axes[0, 0].set_ylabel(r'Lateral force $F_t$ (nN)')
    axes[0, 0].legend(loc='upper right', fontsize=7.0, frameon=False, ncol=2)
    panel_label(axes[0, 0], r'$F_t(\phi)$', x=0.04, y=0.93)

    axes[0, 1].axhline(0, color='#666666', ls=':', lw=0.8)
    axes[0, 1].set_ylabel(r'Normal force $F_z$ (nN)')
    panel_label(axes[0, 1], r'$F_z(\phi)$', x=0.04, y=0.93)

    axes[1, 0].axhline(0, color='#888888', ls=':', lw=0.6)
    axes[1, 0].axvline(0, color='#888888', ls=':', lw=0.6)
    axes[1, 0].set_xlabel(r'Upward displacement $z - z_c$ (Å)')
    axes[1, 0].set_ylabel(r'Normal force $F_z$ (nN)')
    panel_label(axes[1, 0], r'$F_z$--$z$ loop', x=0.04, y=0.93)
    minor(axes[1, 0], 2, 2)

    axes[1, 1].axhline(0, color='#666666', ls=':', lw=0.8)
    axes[1, 1].set_xlabel(r'Nominal penetration $h$ (Å)')
    axes[1, 1].set_ylabel(r'Stroke force difference $F_t^- - F_t^+$ (nN)')
    axes[1, 1].set_xlim(4.5, 15.5)
    axes[1, 1].set_ylim(-12, 42)
    axes[1, 1].set_yticks([-10, 0, 10, 20, 30, 40])
    panel_label(axes[1, 1], r'Matched-depth $\Delta F_t$', x=0.04, y=0.94)
    minor(axes[1, 1], 2, 2)

    save(fig, 'phase_response')
    write_csv(DATA / 'phase_profiles.csv', profiles)

    # Individual subfigure exports for Figure 5
    fig_a, ax_a = plt.subplots(figsize=(3.4, 2.3), layout='constrained')
    for item in plot_items:
        ax_a.plot(phase / np.pi, item['ft_mean'], label=item['label'], color=item['color'], lw=1.3)
        ax_a.fill_between(phase / np.pi, item['ft_mean'] - item['spread_ft'], item['ft_mean'] + item['spread_ft'], color=item['color'], alpha=0.12)
    ax_a.axvline(1, color='#666666', ls='--', lw=0.8)
    ax_a.set_xlabel(r'Phase $\phi/\pi$')
    ax_a.set_xlim(0, 2)
    ax_a.set_xticks([0, .5, 1, 1.5, 2])
    ax_a.set_ylabel(r'Lateral force $F_t$ (nN)')
    ax_a.text(0.25, 0.05, r'$\leftarrow$ Downstroke', transform=ax_a.transAxes,
              ha='center', va='bottom', fontsize=7.5, color='#555555')
    ax_a.text(0.75, 0.05, r'Upstroke $\rightarrow$', transform=ax_a.transAxes,
              ha='center', va='bottom', fontsize=7.5, color='#555555')
    ax_a.legend(loc='upper right', fontsize=6.8, frameon=False, ncol=2)
    minor(ax_a, 2, 2)
    save(fig_a, 'phase_response_a')

    fig_b, ax_b = plt.subplots(figsize=(3.4, 2.3), layout='constrained')
    for item in plot_items:
        ax_b.plot(phase / np.pi, item['fz_mean'], label=item['label'], color=item['color'], lw=1.3)
        ax_b.fill_between(phase / np.pi, item['fz_mean'] - item['spread_fz'], item['fz_mean'] + item['spread_fz'], color=item['color'], alpha=0.12)
    ax_b.axvline(1, color='#666666', ls='--', lw=0.8)
    ax_b.axhline(0, color='#666666', ls=':', lw=0.8)
    ax_b.set_xlabel(r'Phase $\phi/\pi$')
    ax_b.set_xlim(0, 2)
    ax_b.set_xticks([0, .5, 1, 1.5, 2])
    ax_b.set_ylabel(r'Normal force $F_z$ (nN)')
    minor(ax_b, 2, 2)
    save(fig_b, 'phase_response_b')

    fig_c, ax_c = plt.subplots(figsize=(3.4, 2.3), layout='constrained')
    for item in plot_items:
        ax_c.plot(item['z_disp'], item['fz_mean'], color=item['color'], lw=1.3, label=item['label'])
    ax_c.axhline(0, color='#888888', ls=':', lw=0.6)
    ax_c.axvline(0, color='#888888', ls=':', lw=0.6)
    ax_c.set_xlabel(r'Upward displacement $z - z_c$ (Å)')
    ax_c.set_ylabel(r'Normal force $F_z$ (nN)')
    minor(ax_c, 2, 2)
    save(fig_c, 'phase_response_c')

    fig_d, ax_d = plt.subplots(figsize=(3.4, 2.3), layout='constrained')
    for item in plot_items:
        ax_d.plot(item['depth'], item['fmean_diff'], color=item['color'], lw=1.3, label=item['label'])
    ax_d.axhline(0, color='#666666', ls=':', lw=0.8)
    ax_d.set_xlabel(r'Nominal penetration $h$ (Å)')
    ax_d.set_ylabel(r'Stroke force difference $F_t^- - F_t^+$ (nN)')
    ax_d.set_xlim(4.5, 15.5)
    ax_d.set_ylim(-12, 42)
    ax_d.set_yticks([-10, 0, 10, 20, 30, 40])
    minor(ax_d, 2, 2)
    save(fig_d, 'phase_response_d')

    apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), layout='constrained')

    base_r1_p = groups['Baseline'][0]['total_power_uW']
    base_r2_p = groups['Baseline'][1]['total_power_uW']
    base_r1_ft = groups['Baseline'][0]['Ft_mean_nN']
    base_r2_ft = groups['Baseline'][1]['Ft_mean_nN']

    # Panel (a): Trade-off curve
    ax = axes[0]
    ax.axhline(1.0, color='#78909c', ls='--', lw=0.9, zorder=1)
    ax.text(44.0, 0.98, 'Break-even',
            fontsize=7.2, color='#546e7a', ha='right', va='top', style='italic')

    for i, (label, *_) in enumerate(CASES):
        r1 = groups[label][0]
        r2 = groups[label][1]
        col = COLORS[i]

        x1 = r1['paired_reduction_percent']
        y1 = r1['total_power_uW'] / base_r1_p
        x2 = r2['paired_reduction_percent']
        y2 = r2['total_power_uW'] / base_r2_p

        xm = (x1 + x2) / 2.0
        ym = (y1 + y2) / 2.0
        xs = abs(x1 - x2) / np.sqrt(2)
        ys = abs(y1 - y2) / np.sqrt(2)

        # Tie line
        ax.plot([x1, x2], [y1, y2], color='#cfd8dc', lw=1.2, zorder=2)
        ax.scatter(x1, y1, color='#1f4e9c', marker='o', s=26, zorder=3, alpha=0.9, edgecolors='none')
        ax.scatter(x2, y2, color='#c0392b', marker='s', s=24, zorder=3, alpha=0.9, edgecolors='none')
        ax.errorbar(xm, ym, xerr=xs, yerr=ys, fmt='none', ecolor='#37474f', elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
        ax.scatter(xm, ym, color=col, marker='D', s=36, edgecolors='k', linewidths=0.7, zorder=5)

    ax.text(0.0, 0.93, 'Baseline', fontsize=7.5, fontweight='bold', ha='center', va='top', color='#333333')
    ax.text(4.0, 1.09, 'A10B05', fontsize=7.5, fontweight='bold', ha='left', va='bottom', color='#1b7837')
    ax.text(7.5, 1.48, 'A10B20', fontsize=7.5, fontweight='bold', ha='right', va='bottom', color='#2166ac')
    ax.text(17.55, 1.15, 'A25B10', fontsize=7.5, fontweight='bold', ha='left', va='top', color='#9970ab')
    ax.text(25.0, 1.42, 'A25B20', fontsize=7.5, fontweight='bold', ha='left', va='center', color='#d6604d')
    ax.text(39.9, 1.86, 'A50B20', fontsize=7.5, fontweight='bold', ha='right', va='top', color='#762a83')

    ax.set_xlabel(r'Paired lateral force reduction $R$ (%)')
    ax.set_ylabel(r'Normalized mechanical input $P_{\mathrm{in}} / P_{\mathrm{baseline}}$')
    ax.set_xlim(-4, 46)
    ax.set_ylim(0.88, 2.05)
    minor(ax, 2, 2)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f4e9c', markersize=5.5, label='Realization 1'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#c0392b', markersize=5.5, label='Realization 2'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#757575', markeredgecolor='k', markersize=5.5, label=r'Condition mean $\pm s$'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7.0, frameon=True,
              facecolor='white', edgecolor='#e0e0e0', framealpha=0.95, bbox_to_anchor=(0.02, 0.88))
    panel_label(ax, r'Force–power trade-off', x=0.035, y=0.94)

    # Panel (b): Component power balance
    ax2 = axes[1]
    cond_labels = [c[0] for c in CASES[1:]]
    x_pos = np.arange(len(cond_labels))
    width = 0.35

    d_px_means, d_px_sds = [], []
    pz_means, pz_sds = [], []
    ratios = []

    for label in cond_labels:
        r1 = groups[label][0]
        r2 = groups[label][1]
        d1 = (base_r1_ft - r1['Ft_mean_nN']) * 0.03
        d2 = (base_r2_ft - r2['Ft_mean_nN']) * 0.03
        dm = (d1 + d2) / 2.0
        ds = abs(d1 - d2) / np.sqrt(2)
        d_px_means.append(dm)
        d_px_sds.append(ds)

        z1 = r1['Pz_uW']
        z2 = r2['Pz_uW']
        zm = (z1 + z2) / 2.0
        zs = abs(z1 - z2) / np.sqrt(2)
        pz_means.append(zm)
        pz_sds.append(zs)
        ratios.append(zm / dm)

    ax2.bar(x_pos - width/2, d_px_means, width, yerr=d_px_sds,
            capsize=2.5, color='#2e7d32', alpha=0.85, edgecolor='k', lw=0.6,
            label=r'Lateral power saved $\Delta P_x$')
    ax2.bar(x_pos + width/2, pz_means, width, yerr=pz_sds,
            capsize=2.5, color='#d35400', alpha=0.85, edgecolor='k', lw=0.6,
            label=r'Vertical drive power $P_z$')

    for i in range(len(cond_labels)):
        y_top = max(d_px_means[i] + d_px_sds[i], pz_means[i] + pz_sds[i])
        ax2.text(x_pos[i] + width/2, y_top + 0.06,
                 f'{ratios[i]:.1f}' + r'$\times$',
                 ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#c0392b')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cond_labels, rotation=0, ha='center', fontsize=8.0)
    ax2.set_ylabel(r'Mechanical power component ($\mu$W)')
    ax2.set_ylim(0, 2.75)
    minor(ax2, 1, 2)
    ax2.legend(loc='upper left', fontsize=7.0, frameon=True,
               facecolor='white', edgecolor='#e0e0e0', framealpha=0.95,
               bbox_to_anchor=(0.02, 0.88))
    panel_label(ax2, r'Component power balance', x=0.035, y=0.94)

    save(fig, 'force_power')

    # Individual subfigure exports for Figure 6
    fig_fa, ax_fa = plt.subplots(figsize=(3.4, 2.4), layout='constrained')
    ax_fa.axhline(1.0, color='#78909c', ls='--', lw=0.9, zorder=1)
    ax_fa.text(44.0, 0.98, 'Break-even',
               fontsize=6.8, color='#546e7a', ha='right', va='top', style='italic')

    for i, (label, *_) in enumerate(CASES):
        r1 = groups[label][0]
        r2 = groups[label][1]
        col = COLORS[i]

        x1 = r1['paired_reduction_percent']
        y1 = r1['total_power_uW'] / base_r1_p
        x2 = r2['paired_reduction_percent']
        y2 = r2['total_power_uW'] / base_r2_p

        xm = (x1 + x2) / 2.0
        ym = (y1 + y2) / 2.0
        xs = abs(x1 - x2) / np.sqrt(2)
        ys = abs(y1 - y2) / np.sqrt(2)

        ax_fa.plot([x1, x2], [y1, y2], color='#cfd8dc', lw=1.2, zorder=2)
        ax_fa.scatter(x1, y1, color='#1f4e9c', marker='o', s=24, zorder=3, alpha=0.9, edgecolors='none')
        ax_fa.scatter(x2, y2, color='#c0392b', marker='s', s=22, zorder=3, alpha=0.9, edgecolors='none')
        ax_fa.errorbar(xm, ym, xerr=xs, yerr=ys, fmt='none', ecolor='#37474f', elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
        ax_fa.scatter(xm, ym, color=col, marker='D', s=34, edgecolors='k', linewidths=0.7, zorder=5)

    ax_fa.text(0.0, 0.93, 'Baseline', fontsize=7.2, fontweight='bold', ha='center', va='top', color='#333333')
    ax_fa.text(4.0, 1.09, 'A10B05', fontsize=7.2, fontweight='bold', ha='left', va='bottom', color='#1b7837')
    ax_fa.text(7.5, 1.48, 'A10B20', fontsize=7.2, fontweight='bold', ha='right', va='bottom', color='#2166ac')
    ax_fa.text(17.55, 1.15, 'A25B10', fontsize=7.2, fontweight='bold', ha='left', va='top', color='#9970ab')
    ax_fa.text(25.0, 1.42, 'A25B20', fontsize=7.2, fontweight='bold', ha='left', va='center', color='#d6604d')
    ax_fa.text(39.9, 1.86, 'A50B20', fontsize=7.2, fontweight='bold', ha='right', va='top', color='#762a83')

    ax_fa.set_xlabel(r'Paired lateral force reduction $R$ (%)')
    ax_fa.set_ylabel(r'Normalized mechanical input $P_{\mathrm{in}} / P_{\mathrm{baseline}}$')
    ax_fa.set_xlim(-4, 46)
    ax_fa.set_ylim(0.88, 2.05)
    minor(ax_fa, 2, 2)
    ax_fa.legend(handles=legend_elements, loc='upper left', fontsize=6.5, frameon=True,
                 facecolor='white', edgecolor='#e0e0e0', framealpha=0.95, bbox_to_anchor=(0.02, 0.88))
    save(fig_fa, 'force_power_a')

    fig_fb, ax_fb = plt.subplots(figsize=(3.4, 2.4), layout='constrained')
    ax_fb.bar(x_pos - width/2, d_px_means, width, yerr=d_px_sds,
              capsize=2.5, color='#2e7d32', alpha=0.85, edgecolor='k', lw=0.6,
              label=r'Lateral power saved $\Delta P_x$')
    ax_fb.bar(x_pos + width/2, pz_means, width, yerr=pz_sds,
              capsize=2.5, color='#d35400', alpha=0.85, edgecolor='k', lw=0.6,
              label=r'Vertical drive power $P_z$')

    for i in range(len(cond_labels)):
        y_top = max(d_px_means[i] + d_px_sds[i], pz_means[i] + pz_sds[i])
        ax_fb.text(x_pos[i] + width/2, y_top + 0.06,
                   f'{ratios[i]:.1f}' + r'$\times$',
                   ha='center', va='bottom', fontsize=7.2, fontweight='bold', color='#c0392b')

    ax_fb.set_xticks(x_pos)
    ax_fb.set_xticklabels(cond_labels, rotation=0, ha='center', fontsize=7.8)
    ax_fb.set_ylabel(r'Mechanical power component ($\mu$W)')
    ax_fb.set_ylim(0, 2.75)
    minor(ax_fb, 1, 2)
    ax_fb.legend(loc='upper left', fontsize=6.5, frameon=True,
                 facecolor='white', edgecolor='#e0e0e0', framealpha=0.95,
                 bbox_to_anchor=(0.02, 0.88))
    save(fig_fb, 'force_power_b')

    save(fig, 'force_power')
    controls(args, series)
    structure_check(args.offline)
    layouts = {
        'force_rows': (
            r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}\toprule' + '\n' +
            r'Case & $\overline{F}_t$ (nN) & $\overline{F}_z$ (nN) & Tensile duty (\%) & Paired reduction (\%)\\\midrule',
            r'\bottomrule' + '\n' + r'\end{tabular*}'
        ),
        'stroke_rows': (
            r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}\toprule' + '\n' +
            r'Case & Downstroke $\overline{F}_t^-$ (nN) & Upstroke $\overline{F}_t^+$ (nN) & Attenuation (\%) & $\overline{W}_z$ (eV/cycle) & Complete cycles (R1/R2)\\\midrule',
            r'\bottomrule' + '\n' + r'\end{tabular*}'
        ),
        'power_rows': (
            r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcc@{}}\toprule' + '\n' +
            r'Case & Vertical drive $P_z$ ($\mu$W) & Total input $\overline{P}_{\mathrm{in}}$ ($\mu$W)\\\midrule',
            r'\bottomrule' + '\n' + r'\end{tabular*}'
        ),
        'baseline_rows': (
            r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}\toprule' + '\n' +
            r'Realization & Evaluation interval (nm) & Mean $\overline{F}_t$ (nN) & Temporal $\sigma_t$ (nN) & Mean $\overline{F}_z$ (nN) & Temporal $\sigma_z$ (nN)\\\midrule',
            r'\bottomrule' + '\n' + r'\end{tabular*}'
        ),
        'control_rows': (
            r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcccc@{}}\toprule' + '\n' +
            r'Condition & Rep. & $\overline{F}_t$ (nN) & Tensile (\%) & $\overline{P}_{\mathrm{in}}$ ($\mu$W)\\\midrule',
            r'\bottomrule' + '\n' + r'\end{tabular*}'
        ),
    }
    for name, (header, footer) in layouts.items():
        path = GEN / f'{name}.tex'
        body = path.read_text()
        if r'\begin{tabular' in body:
            clean_lines = [line for line in body.splitlines() if not (
                line.startswith(r'\begin{tabular') or
                line.startswith(r'\end{tabular') or
                line.startswith(r'\toprule') or
                line.startswith(r'\bottomrule') or
                line.startswith(r'\midrule') or
                line.startswith('Case &') or
                line.startswith('Realization &') or
                line.startswith('Simulation condition &') or
                line.startswith('Control &')
            )]
            body = '\n'.join(clean_lines).strip() + '\n'
        path.write_text(f'{header}\n{body}{footer}\n')
    print(json.dumps(stats, indent=2))

if __name__=='__main__': main()
