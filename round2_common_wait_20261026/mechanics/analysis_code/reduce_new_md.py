#!/usr/bin/env python3
"""Reduce a verified completed control without promoting it into the paper."""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def linear_cos_integral(t, force, omega, origin):
    """Exact integral of piecewise-linear force times an analytic cosine."""
    t=np.asarray(t,dtype=float); force=np.asarray(force,dtype=float)
    a=t[:-1]-origin;b=t[1:]-origin
    slope=np.diff(force)/np.diff(t);intercept=force[:-1]-slope*a
    def primitive(u):
        return intercept*np.sin(omega*u)/omega+slope*(u*np.sin(omega*u)/omega+np.cos(omega*u)/omega**2)
    return float((primitive(b)-primitive(a)).sum())


def negative_duration(t,f):
    total=0.
    for a,b,fa,fb in zip(t[:-1],t[1:],f[:-1],f[1:]):
        if fa<=0 and fb<=0 and min(fa,fb)<0:total+=b-a
        elif fa*fb<0:
            crossing=a+(b-a)*(-fa)/(fb-fa)
            total+=crossing-a if fa<0 else b-crossing
    return total


def reduce_case(run, out, legacy_case=None):
    run=Path(run);out=Path(out)
    launch_path=run/'launch.json' if legacy_case is None else run/'metadata/launch.json'
    log_path=run/'lammps.log' if legacy_case is None else run/'logs/lammps.log'
    launch=json.loads(launch_path.read_text());case=launch['case'] if legacy_case is None else legacy_case
    if launch.get('smoke',False):raise ValueError('Pilot cannot enter scientific analysis')
    rows={};header=None
    for line in log_path.read_text().splitlines():
        fields=line.split()
        if fields[:2]==['Step','Time']:header=fields
        elif header and 'v_tool_x' in header and len(fields)==len(header):
            try:values=list(map(float,fields))
            except ValueError:continue
            if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite record')
            rows[int(values[0])]=dict(zip(header,values))
    if not rows:raise ValueError('No force records')
    series=[rows[s] for s in sorted(rows)]
    t=np.array([r['Time'] for r in series]);x=np.array([r['v_tool_x'] for r in series]);
    ft=np.array([r['v_fx'] for r in series])*1.602176634
    fz=np.array([r['v_fz'] for r in series])*1.602176634
    travel=(x[0]-x)/10.
    translating=np.concatenate(([True],np.diff(travel)>1e-10))
    t=t[translating];ft=ft[translating];fz=fz[translating];travel=travel[translating]
    if travel[0]>4 or travel[-1]<16:raise ValueError('Common travel window is not covered')
    lo=float(np.interp(4.,travel,t));hi=float(np.interp(16.,travel,t))
    grid=np.concatenate(([lo],t[(t>lo)&(t<hi)],[hi]))
    ft=np.interp(grid,t,ft);fz=np.interp(grid,t,fz)
    duration=hi-lo;mean_ft=float(np.trapezoid(ft,grid)/duration)
    v=case['variables'];lateral=.1*v['vs']*mean_ft;vertical=0.;cycles=[]
    stroke=None
    if case['oscillatory']:
        omega=2*math.pi/case['period_ps'];A=v['alpha']*v['h0']
        origin=v['eq_steps_pre']*v['dt_ps']+v['wait_steps']*v['dt_ps']
        vertical=-.1*A*omega*linear_cos_integral(grid,fz,omega,origin)/duration
        first=math.ceil((lo-origin)/case['period_ps']);last=math.floor((hi-origin)/case['period_ps'])
        for k in range(first,last):
            aa=origin+k*case['period_ps'];bb=aa+case['period_ps']
            tt=np.concatenate(([aa],grid[(grid>aa)&(grid<bb)],[bb]));ff=np.interp(tt,grid,fz)
            work=-A*omega*linear_cos_integral(tt,ff,omega,origin)/1.602176634
            cycles.append(work)
        # Add velocity zeros before integrating force separately on the two strokes.
        k0=math.floor(((lo-origin)*omega-math.pi/2)/math.pi)
        k1=math.ceil(((hi-origin)*omega-math.pi/2)/math.pi)
        zeros=[origin+(math.pi/2+k*math.pi)/omega for k in range(k0,k1+1)]
        split=np.array(sorted(set(grid.tolist()+[z for z in zeros if lo<z<hi])))
        loads=np.interp(split,grid,ft);dt=np.diff(split);mid=(split[:-1]+split[1:])/2
        up=np.cos(omega*(mid-origin))>0;ints=dt*(loads[:-1]+loads[1:])/2
        up_mean=float(ints[up].sum()/dt[up].sum());down_mean=float(ints[~up].sum()/dt[~up].sum())
        stroke={'up_Ft_nN':up_mean,'down_Ft_nN':down_mean,'attenuation_percent':100*(1-up_mean/down_mean)}
    result={'case_id':case['case_id'],'group':case['group'],'preparation':case['preparation'],
            'condition':case['condition'],'status':'REDUCED_PENDING_STRUCTURAL_AND_PAIR_QA',
            'travel_window_nm':[4,16],'window_time_ps':[lo,hi],'mean_Ft_nN':mean_ft,
            'mean_Fz_nN':float(np.trapezoid(fz,grid)/duration),
            'tensile_duty_percent':100*negative_duration(grid,fz)/duration,'stroke':stroke,
            'lateral_power_uW':lateral,'vertical_power_uW':vertical,'net_input_power_uW':lateral+vertical,
            'complete_cycle_count':len(cycles),'cycle_work_eV':cycles,
            'quadrature':'Exact integration of piecewise-linear signed force with analytic normal velocity',
            'log_sha256':digest(log_path),'launch_sha256':digest(launch_path),
            'independent_statistical_units':1,'publication_ready':False,
            'limits':['Single preparation result, not an ensemble uncertainty','Matched baseline and structural QA required','Tool-interface net input is not irreversible dissipation']}
    if legacy_case is not None:
        result['historical_case_id']=launch.get('case_id',case['case_id'])
    out.mkdir(parents=True,exist_ok=True);(out/(case['case_id']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import sys
    print(json.dumps(reduce_case(sys.argv[1],sys.argv[2])))
