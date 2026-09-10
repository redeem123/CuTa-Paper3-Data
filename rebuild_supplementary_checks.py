#!/usr/bin/env python3
"""Generate unrounded window and quadrature checks from packaged force samples."""
from pathlib import Path
import json
import csv
import numpy as np
from scipy.integrate import simpson
from rebuild_analysis import CASES, interval, moments
ROOT=Path(__file__).resolve().parent

def write(name,rows):
    with (ROOT/'data'/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def windows():
    runs=[];rows=[];tex=[]
    for label,stem,amp,beta in CASES:
        for start in [4.,6.,8.]:
            pair=[]
            for rep in [1,2]:
                case=f'{stem}_R{rep:02d}'
                a=np.genfromtxt(ROOT/'data'/f'{case}.csv',delimiter=',',names=True)
                b=np.genfromtxt(ROOT/'data'/f'scratch_base_R{rep:02d}.csv',delimiter=',',names=True)
                m=json.loads((ROOT/'data'/f'{case}.json').read_text());t=a['time_ps'];lo=t[0]+start/.03;hi=t[0]+16/.03
                ft=moments(t,a['Ft_nN'],lo,hi)[0]
                bf=moments(b['time_ps'],b['Ft_nN'],b['time_ps'][0]+start/.03,b['time_ps'][0]+16/.03)[0]
                tt,ff=interval(t,a['Fz_nN'],lo,hi);om=2*np.pi*m['frequency_per_ps']
                pz=float(-np.trapezoid(ff*om*amp*np.cos(om*tt+m['phase_offset']),tt)/(hi-lo)*.1)
                vals=dict(Ft=ft,Red=100*(1-ft/bf),Pz=pz,Pin=.03*ft+pz)
                runs.append(dict(case=case,condition=label,realization=rep,window_start_nm=start,window_end_nm=16.,window_start_ps=lo,window_end_ps=hi,**vals));pair.append(vals)
            r=dict(condition=label,window_nm=f'{start:.1f}--16.0',duration_ps=(16-start)/.03)
            cells=[label,r['window_nm']]
            for k in ['Ft','Red','Pz','Pin']:
                v=[p[k] for p in pair];r[k+'_mean']=float(np.mean(v));r[k+'_std']=float(np.std(v,ddof=1))
                cells.append('N/A' if k=='Red' and label=='Baseline' else f"${r[k+'_mean']:.2f} \\pm {r[k+'_std']:.2f}$")
            rows.append(r);tex.append(' & '.join(cells)+r' \\')
    write('supp_window_runs.csv',runs);write('supp_window_sensitivity.csv',rows)
    (ROOT/'generated/supp_window_rows.tex').write_text('\n'.join(tex)+'\n'+r'\bottomrule'+'\n')

def quadrature():
    rows=[];tex=[]
    for label,stem,amp,beta in CASES[1:]:
        case=stem+'_R01';m=json.loads((ROOT/'data'/f'{case}.json').read_text())
        a=np.genfromtxt(ROOT/'data'/f'{case}.csv',delimiter=',',names=True);t=a['time_ps']
        lo=t[0]+4/.03;hi=t[0]+16/.03;tt,ff=interval(t,a['Fz_nN'],lo,hi);om=2*np.pi*m['frequency_per_ps']
        y=ff*om*amp*np.cos(om*tt+m['phase_offset'])
        trap=float(-np.trapezoid(y,tt)/(hi-lo)*.1);simp=float(-simpson(y,x=tt)/(hi-lo)*.1)
        r=dict(condition=label,period_ps=1/m['frequency_per_ps'],dt_sample_ps=m['sample_interval_ps'],pts_per_cycle=1/m['frequency_per_ps']/m['sample_interval_ps'],Pz_trap_uW=trap,Pz_simp_uW=simp,diff_pct=100*abs(simp-trap)/trap)
        for stride in [2,4]:
            # Keep identical integration bounds when coarsening the recorded grid.
            ix=np.unique(np.r_[np.arange(0,len(tt),stride),len(tt)-1])
            coarse=float(-np.trapezoid(y[ix],tt[ix])/(hi-lo)*.1)
            r[f'downsample_{stride}x_pct']=100*abs(coarse-trap)/trap
        rows.append(r)
        tex.append(f"{label} & {trap:.4f} & {simp:.4f} & {r['diff_pct']:.3f} & {r['downsample_2x_pct']:.3f} & {r['downsample_4x_pct']:.3f}"+r' \\')
    write('supp_quadrature_downsampling.csv',rows)
    (ROOT/'generated/supp_quadrature_rows.tex').write_text('\n'.join(tex)+'\n'+r'\bottomrule'+'\n')
    print('Maximum differences (%):',*[max(r[k] for r in rows) for k in ['diff_pct','downsample_2x_pct','downsample_4x_pct']])

if __name__=='__main__':windows();quadrature()
