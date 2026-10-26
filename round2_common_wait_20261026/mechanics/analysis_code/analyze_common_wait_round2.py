#!/usr/bin/env python3
"""Qualify and reduce the eighteen common-wait histories without editing papers."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from primary_reference import raw_blocks, analyze, interval, moments, write_csv
from reduce_new_md import linear_cos_integral
from complete_common_wait import independent_integral

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
OUT = R / 'analysis/a10_common_wait_20261020'
SRC = OUT / 'source_records'
CASES = [('Baseline', 'scratch_base', 0., 0.), ('A10B05', 'S1', 1., .5),
         ('A10B20', 'S2', 1., 2.), ('A25B10', 'S4', 2.5, 1.),
         ('A25B20', 'S5', 2.5, 2.), ('A50B20', 'S6', 5., 2.)]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def stats(vals):
    return dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)))

def log_series(run, case, collection=None):
    launch = json.loads((run/'launch.json').read_text())
    assert launch['case'] == case and not launch['smoke']
    name=case['case_id']; v=case['variables']; paths=[run/'lammps.log']; join=None
    blocks=raw_blocks(paths[0])
    moving=[(h,a) for h,a in blocks if np.ptp(a[:,h.index('v_tool_x')])>1]
    assert len(moving)==1,(name,len(moving))
    h,a=moving[0]
    for bh,ba in blocks:
        if np.ptp(ba[:,bh.index('v_tool_x')])<=1:
            assert np.max(np.abs(ba[:,[bh.index('v_fx'),bh.index('v_fz')]]))==0
    if name=='P3rev_wait_A10B20_R01':
        op=SRC/'operations/checkpoint_migration_20261021'
        segment=json.loads((op/'segment_manifest.json').read_text())
        assert sha(paths[0])==segment['old_segment']['closed_artifact_sha256']['lammps.log']
        cont=SRC/'continuations'/name/'dgx01_segment01'
        cl=json.loads((cont/'launch.json').read_text());assert cl['scientific_variables']==v
        assert cl['restart_sha256']==segment['snapshot_sha256']
        paths.append(cont/'lammps.log'); hb,b=raw_blocks(paths[-1])[-1];assert hb==h
        # The approved ledger excludes run0 and start rows at400000; retain the original cadence.
        prefix=a[a[:,0]<=400000]; suffix=b[b[:,0]>400000]
        assert prefix[-1,0]==399984 and suffix[0,0]==400088
        diag=json.loads((op/'qualified_existing_diagnostics.json').read_text())
        err={}
        for step,record in diag['original_thermo_rows'].items():
            row=suffix[suffix[:,0]==int(step)][0]
            for key in ['Time','v_tool_x','v_tool_z','v_phase','v_fx','v_fy','v_fz']:
                delta=abs(float(row[h.index(key)])-record[key])
                tol=max(1e-8,2*10**(np.floor(np.log10(abs(record[key]))) - 7)) if record[key] else 1e-8
                assert delta<=tol,(key,step,delta,tol)
                err[key]=max(err.get(key,0),delta)
        join=dict(prefix_last_step=int(prefix[-1,0]),suffix_first_step=int(suffix[0,0]),
                  excluded_original_after_checkpoint=int(np.sum(a[:,0]>400000)),
                  excluded_continuation_start_rows=int(np.sum(b[:,0]<=400000)),
                  first_five_record_errors=err,selection='Original<=400000; continuation>400000; run0 excluded')
        a=np.vstack([prefix,suffix])
    d={k:a[:,i] for i,k in enumerate(h)}; t=d['Time']; step=d['Step']
    assert np.isfinite(a).all() and np.all(np.diff(t)>0)
    assert abs(t[0]-161.080)<1e-9 and int(step[-1])==case['expected_final_step']==697747
    assert np.max(np.abs(t-.001*step))<1e-9
    cadence=v.get('thermo_every',50)
    assert np.max(np.diff(step))<=cadence and np.all(step[1:-1]%cadence==0),(name,cadence,np.max(np.diff(step)))
    travel=(d['v_tool_x'][0]-d['v_tool_x'])/10
    xerr=float(np.max(np.abs(travel-.03*(t-t[0])))); assert xerr<1e-5
    amp=v.get('A_case',0) if case['oscillatory'] else 0.;freq=v.get('f_case',0) if amp else 0.
    omega=2*np.pi*freq; theta=np.unwrap(2*np.pi*d['v_phase']) if amp else np.zeros_like(t)
    offset=float(np.median(theta-omega*t)) if amp else 0.
    phaseerr=float(np.max(np.abs(theta-(omega*t+offset)))) if amp else 0.
    z=d.get('v_tool_z',np.zeros_like(t)); zerr=float(np.ptp(z-amp*np.sin(theta))) if amp else 0.
    assert phaseerr<1e-5 and zerr<1e-3
    if amp:
        origin=4+v['wait_steps']*.001
        assert abs(np.angle(np.exp(1j*(offset+omega*origin))))<1e-5
    series=dict(time_ps=t,travel_nm=travel,Ft_nN=d['v_fx']*1.602176634,Fz_nN=d['v_fz']*1.602176634,
                z_A=z,theta_rad=theta,vz_A_ps=omega*amp*np.cos(theta))
    meta=dict(case=name,amplitude_A=amp,beta=v.get('beta',0),frequency_per_ps=freq,phase_offset=offset,
              sample_interval_ps=float(np.median(np.diff(t))),structure_sha256=CAMPAIGN['structures'][str(case['preparation'])]['sha256'],
              source_logs=[dict(path=str(p),sha256=sha(p)) for p in paths],source_launch_sha256=sha(run/'launch.json'),
              translation_start_ps=float(t[0]),wait_ps=float(t[0]-4),x_error_nm=xerr,phase_error_rad=phaseerr,
              sinusoid_error_A=zerr,sample_count=len(t),join=join)
    return series,meta

def main():
    global CAMPAIGN
    CAMPAIGN=json.loads((R/'simulation/campaign_v4/campaign.json').read_text())
    ac=json.loads((R/'simulation/a10_common_wait_20261020/campaign.json').read_text())
    collect=json.loads((OUT/'collection_receipt.json').read_text())
    for rel,identity in collect['received_scalar_sources'].items():assert sha(SRC/rel)==identity['sha256'],rel
    for rep in (1,2,3):
        remote=ac['source_stage']+f'/inputs/R{rep:02d}.data'
        assert collect['source_identities'][remote]['sha256']==CAMPAIGN['structures'][str(rep)]['sha256']
    for name in ['oscillatory.lmp','CuTa_LJ15_2014.adp.txt']:
        assert collect['source_identities'][ac['source_stage']+'/inputs/'+name]['sha256']==sha(R/'simulation/campaign_v4/inputs'/name)
    prior=json.loads((R/'analysis/common_wait_20261002/report.json').read_text())
    pr={x['case_id']:x for x in prior['sources']}
    oldrows={x['case_id']:x for x in csv.DictReader((R/'analysis/common_wait_20261002/run_metrics.csv').open())}
    controls=ROOT/'merged_workspaces/CuTa-results/validation/paper3_revision_v1_20261001_v4/runs'
    selected={}; rows=[];cycles=[];sources=[];replay=[];quad=[]
    (OUT/'series').mkdir(exist_ok=True)
    for condition,stem,amp,beta in CASES:
        for rep in (1,2,3):
            if condition in ('A25B10','A50B20'):
                name=f'{stem}_R{rep:02d}';m=json.loads((R/'data'/f'{name}.json').read_text())
                assert sha(R/'data'/f'{name}.csv')==m['samples_sha256']==pr[name]['samples_sha256']
                assert m['log_sha256']==pr[name]['log_sha256']
                assert m['structure_sha256']==CAMPAIGN['structures'][str(rep)]['sha256']
                ar=np.genfromtxt(R/'data'/f'{name}.csv',names=True,delimiter=',');d={k:ar[k] for k in ar.dtype.names}
                assert abs(d['time_ps'][0]-161.080)<1e-9
            else:
                name=f'P3rev_wait_{"Base" if condition=="Baseline" else condition}_R{rep:02d}'
                case=next(x for x in (ac['cases'] if condition.startswith('A10') else CAMPAIGN['cases']) if x['case_id']==name)
                run=(SRC/'runs' if condition.startswith('A10') else controls)/name
                if name in pr:
                    assert sha(run/'lammps.log')==pr[name]['log_sha256'] and sha(run/'launch.json')==pr[name]['launch_sha256']
                d,m=log_series(run,case,collect)
            row,cc=analyze(d,m);row.update(condition=condition,realization=rep)
            if name in oldrows:
                mapping={'Ft_mean_nN':'mean_Ft_nN','Fz_mean_nN':'mean_Fz_nN','Pz_uW':'vertical_power_uW','total_power_uW':'net_input_power_uW'}
                delta={k:abs(row[k]-float(oldrows[name][v])) for k,v in mapping.items()};assert max(delta.values())<1e-10,(name,delta)
                replay.append(dict(case=name,max_error=max(delta.values())))
            selected[condition,rep]=(d,m);rows.append(row);cycles.extend(cc);sources.append(m)
            write_csv(OUT/'series'/f'{name}.csv',[dict(zip(d,x)) for x in zip(*d.values())])
            (OUT/'series'/f'{name}.json').write_text(json.dumps(m,indent=2)+'\n')
            if amp:
                om=2*np.pi*m['frequency_per_ps'];origin=-m['phase_offset']/om
                for lo_nm in (4,6,8):
                    tt,ff=interval(d['time_ps'],d['Fz_nN'],d['time_ps'][0]+lo_nm/.03,d['time_ps'][0]+16/.03)
                    exact=linear_cos_integral(tt,ff,om,origin);gauss=independent_integral(tt,ff,om,origin)
                    error=abs(-.1*amp*om*(exact-gauss)/(tt[-1]-tt[0]));assert error<1e-9
                    quad.append(dict(case=name,window_start_nm=lo_nm,error_uW=error))
                for c in cc:
                    tt,ff=interval(d['time_ps'],d['Fz_nN'],c['start_ps'],c['end_ps'])
                    error=abs(c['work_eV']+amp*om*independent_integral(tt,ff,om,origin)/1.602176634);assert error<1e-8
    bases={x['realization']:x for x in rows if x['condition']=='Baseline'}
    for row in rows:
        b=bases[row['realization']]
        row['paired_reduction_percent']=100*(1-row['Ft_mean_nN']/b['Ft_mean_nN'])
        row['paired_input_increase_percent']=100*(row['total_power_uW']/b['total_power_uW']-1)
    summary={c:{k:stats([x[k] for x in rows if x['condition']==c]) for k in
               ['Ft_mean_nN','Fz_mean_nN','tensile_percent','Pz_uW','total_power_uW','paired_reduction_percent','paired_input_increase_percent','stroke_reduction_percent']} for c,*_ in CASES}
    contrasts=[]
    for rep in (1,2,3):
        rr={x['condition']:x for x in rows if x['realization']==rep}
        for a,b in [('A10B05','A10B20'),('A25B10','A25B20'),('A25B10','A50B20')]:
            contrasts.append(dict(first=a,second=b,realization=rep,force_reduction_percent=100*(1-rr[b]['Ft_mean_nN']/rr[a]['Ft_mean_nN']),input_increase_percent=100*(rr[b]['total_power_uW']/rr[a]['total_power_uW']-1)))
    window=[]
    for c,stem,amp,beta in CASES:
        for rep in (1,2,3):
            d,m=selected[c,rep];base,bm=selected['Baseline',rep]
            for lo_nm in (4,6,8):
                lo=161.080+lo_nm/.03;hi=161.080+16/.03
                ft=moments(d['time_ps'],d['Ft_nN'],lo,hi)[0];bf=moments(base['time_ps'],base['Ft_nN'],lo,hi)[0]
                tt,ff=interval(d['time_ps'],d['Fz_nN'],lo,hi);om=2*np.pi*m['frequency_per_ps']
                pz=-.1*amp*om*linear_cos_integral(tt,ff,om,-m['phase_offset']/om)/(hi-lo) if amp else 0
                window.append(dict(condition=c,realization=rep,start_nm=lo_nm,Ft=ft,Red=100*(1-ft/bf),Pz=pz,Pin=.03*ft+pz,paired_input_increase_percent=100*((.03*ft+pz)/(.03*bf)-1)))
    write_csv(OUT/'run_metrics.csv',rows);write_csv(OUT/'complete_cycle_work.csv',cycles);write_csv(OUT/'contrasts.csv',contrasts);write_csv(OUT/'window_runs.csv',window)
    report=dict(status='COMMON_WAIT_18_MECHANICAL_ANALYSIS_VERIFIED',independent_preparations=3,trajectories=18,window_nm=[4,16],translation_start_ps=161.080,wait_ps=157.080,summary=summary,contrasts=contrasts,reused_12_replay=replay,
                maximum_gaussian_error_uW=max(x['error_uW'] for x in quad),all_15_lower_force_higher_input=all(x['paired_reduction_percent']>0 and x['paired_input_increase_percent']>0 for x in rows if x['condition']!='Baseline'),
                all_windows_lower_force_higher_input=all(x['Red']>0 and x['paired_input_increase_percent']>0 for x in window if x['condition']!='Baseline'),
                sources=sources,source_code_sha256=sha(Path(__file__)),manuscript_updated=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','summary','contrasts','maximum_gaussian_error_uW','all_15_lower_force_higher_input','all_windows_lower_force_higher_input']},indent=2))

if __name__=='__main__':main()
