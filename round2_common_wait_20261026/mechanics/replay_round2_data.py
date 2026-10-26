#!/usr/bin/env python3
"""Replay the supplied common-wait force records without MD or repository paths."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'analysis_code'))
from primary_reference import analyze, interval, moments
from reduce_new_md import linear_cos_integral

def rows(name): return list(csv.DictReader((R/'results'/name).open()))
def close(a,b):
    assert np.isclose(a,float(b),rtol=1e-10,atol=1e-8),(a,b)
def main():
    manifest=json.loads((R/'MANIFEST.json').read_text())
    for p,entry in manifest['files'].items():
        assert hashlib.sha256((R/p).read_bytes()).hexdigest()==entry['sha256'],p
    runs=rows('run_metrics.csv'); assert len(runs)==18
    series={}; actual={};checked=0
    for expected in runs:
        case=expected['case'];p=R/'series'/f'{case}.csv'
        a=np.genfromtxt(p,delimiter=',',names=True);d={k:a[k] for k in a.dtype.names}
        m=json.loads((R/'series'/f'{case}.json').read_text())
        assert abs(d['time_ps'][0]-161.080)<1e-9
        assert np.all(np.diff(d['time_ps'])>0)
        row,cycles=analyze(d,m)
        for k,v in row.items():
            if k in expected and isinstance(v,(float,int)): close(v,expected[k]);checked+=1
        row.update(condition=expected['condition'],realization=int(expected['realization']))
        actual[case]=row;series[(row['condition'],row['realization'])]=(d,m)
    base={r['realization']:r for r in actual.values() if r['condition']=='Baseline'}
    for expected in runs:
        r=actual[expected['case']];r['paired_reduction_percent']=100*(1-r['Ft_mean_nN']/base[r['realization']]['Ft_mean_nN'])
        close(r['paired_reduction_percent'],expected['paired_reduction_percent']);checked+=1
    oscillatory=[r for r in actual.values() if r['condition']!='Baseline']
    assert sum(r['paired_reduction_percent']>0 for r in oscillatory)==14
    assert all(r['total_power_uW']>base[r['realization']]['total_power_uW'] for r in oscillatory)
    window_values=[]
    for e in rows('supp_window_runs.csv'):
        cond=e['condition'];rep=int(e['realization']);d,m=series[cond,rep];b,_=series['Baseline',rep]
        start=float(e['window_start_nm']);lo=d['time_ps'][0]+start/.03;hi=d['time_ps'][0]+16/.03
        ft=moments(d['time_ps'],d['Ft_nN'],lo,hi)[0]
        bf=moments(b['time_ps'],b['Ft_nN'],b['time_ps'][0]+start/.03,b['time_ps'][0]+16/.03)[0]
        pz=0.
        if m['amplitude_A']:
            om=2*np.pi*m['frequency_per_ps'];tt,ff=interval(d['time_ps'],d['Fz_nN'],lo,hi)
            pz=-.1*m['amplitude_A']*om*linear_cos_integral(tt,ff,om,-m['phase_offset']/om)/(hi-lo)
        out=dict(Ft=ft,Red=100*(1-ft/bf),Pz=pz,Pin=.03*ft+pz)
        for k,v in out.items():close(v,e[k]);checked+=1
        window_values.append(dict(condition=cond,window_nm=f'{start:g}--16',**out))
    assert len(window_values)==54
    for e in rows('supp_window_sensitivity.csv'):
        group=[v for v in window_values if v['condition']==e['condition'] and v['window_nm']==e['window_nm']];assert len(group)==3
        for k in ['Ft','Red','Pz','Pin']:
            for suffix,v in [('mean',np.mean([g[k] for g in group])),('sd',np.std([g[k] for g in group],ddof=1))]:close(v,e[k+'_'+suffix]);checked+=1
    max_gauss=0.
    for e in rows('supp_quadrature_downsampling.csv'):
        d,m=series[e['condition'],int(e['realization'])];om=2*np.pi*m['frequency_per_ps']
        lo=d['time_ps'][0]+4/.03;tt,ff=interval(d['time_ps'],d['Fz_nN'],lo,lo+400)
        y=ff*om*m['amplitude_A']*np.cos(om*tt+m['phase_offset'])
        trap=-.1*np.trapezoid(y,tt)/400;simp=-.1*simpson(y,x=tt)/400
        exact=-.1*m['amplitude_A']*om*linear_cos_integral(tt,ff,om,-m['phase_offset']/om)/400
        out=dict(Pz_trap_uW=trap,Pz_simp_uW=simp,Pz_exact_linear_force_uW=exact,diff_pct=100*abs(simp-trap)/trap)
        for stride in [2,4]:
            ix=np.unique(np.r_[np.arange(0,len(tt),stride),len(tt)-1]);coarse=-.1*np.trapezoid(y[ix],tt[ix])/400
            out[f'downsample_{stride}x_pct']=100*abs(coarse-trap)/trap
        for k,v in out.items():close(v,e[k]);checked+=1
        xx,ww=np.polynomial.legendre.leggauss(8);dt=np.diff(tt)
        sample=tt[:-1,None]+dt[:,None]*(xx+1)/2
        force=ff[:-1,None]+np.diff(ff)[:,None]*(xx+1)/2
        gauss=-.1*m['amplitude_A']*om*np.sum(dt[:,None]/2*ww*force*np.cos(om*sample+m['phase_offset']))/400
        max_gauss=max(max_gauss,abs(gauss-exact));assert abs(gauss-exact)<1e-9
    figures={}
    for cond in sorted({r['condition'] for r in oscillatory}):
        rr=[r for r in oscillatory if r['condition']==cond]
        vals={'force_reduction_percent':[r['paired_reduction_percent'] for r in rr],
              'normalized_input':[r['total_power_uW']/base[r['realization']]['total_power_uW'] for r in rr],
              'lateral_saving_uW':[.03*(base[r['realization']]['Ft_mean_nN']-r['Ft_mean_nN']) for r in rr],
              'normal_input_uW':[r['Pz_uW'] for r in rr]}
        figures[cond]={k:{'mean':float(np.mean(v)),'sample_SD':float(np.std(v,ddof=1))} for k,v in vals.items()}
    report=dict(status='PASS',trajectories=18,independent_preparations=3,numeric_fields_checked=checked,
        window_records=54,lower_force_pairs=14,higher_input_pairs=15,maximum_gaussian_error_uW=max_gauss,
        adverse_A10B05_R01=actual['P3rev_wait_A10B05_R01'],figure6=figures,
        scope='Force-series reduction, sensitivity and checksums; no MD or structural-field recomputation.')
    (R/'replay_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
