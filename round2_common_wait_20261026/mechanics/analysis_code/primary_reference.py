"""Audited original Paper 3 estimator, isolated for the n=3 revision."""
from pathlib import Path
import csv, hashlib, json, re
import numpy as np
DATA=Path(__file__).resolve().parents[1]/'data'
CONV=1.602176634

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

def snapshot(run, name, amp, beta, offline, log_path=None, launch_path=None):
    path = DATA / f'{name}.csv'
    mp = DATA / f'{name}.json'
    if offline:
        meta=json.loads(mp.read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['samples_sha256'], name
        a = np.genfromtxt(path, delimiter=',', names=True)
        return {k: a[k] for k in a.dtype.names}, meta
    historical_name='ADH100_R03' if name=='S5_R03' else name
    log = Path(log_path) if log_path else run / 'logs/lammps.log'
    text = log.read_text()
    sentinel = re.search(r'^\w*SCRATCH_COMPLETE case='+re.escape(historical_name)+r'\b.*$', text, re.M)
    if not sentinel:
        raise ValueError(f'{name}: raw completion marker absent')
    launch_file=Path(launch_path) if launch_path else run/'metadata/launch.json'
    launch = json.loads(launch_file.read_text())
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
    meta = dict(case=name, historical_case_id=historical_name, amplitude_A=amp, beta=beta, frequency_per_ps=f,
                phase_offset=offset, source_log=str(log),
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                structure_sha256=launch['structure_sha256'],
                source_launch_sha256=hashlib.sha256(launch_file.read_bytes()).hexdigest(),
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

def analyze_legacy(d, m):
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


def analyze(d, m):
    """One estimator for all runs: linear force times analytic normal velocity."""
    from reduce_new_md import linear_cos_integral
    row, cycles = analyze_legacy(d, m)
    if not m['amplitude_A']:
        return row, cycles
    omega = 2*np.pi*m['frequency_per_ps']
    origin = -m['phase_offset']/omega
    for cycle in cycles:
        tt, ff = interval(d['time_ps'], d['Fz_nN'], cycle['start_ps'], cycle['end_ps'])
        cycle['work_eV'] = -m['amplitude_A']*omega*linear_cos_integral(tt, ff, omega, origin)/CONV
    tt, ff = interval(d['time_ps'], d['Fz_nN'], row['window_start_ps'], row['window_end_ps'])
    pz = -.1*m['amplitude_A']*omega*linear_cos_integral(tt, ff, omega, origin)/400.
    row.update(work_eV_cycle=float(np.mean([c['work_eV'] for c in cycles])),
               work_cycle_SD_eV=float(np.std([c['work_eV'] for c in cycles], ddof=1)),
               Pz_uW=float(pz), total_power_uW=.03*row['Ft_mean_nN']+pz)
    return row, cycles
