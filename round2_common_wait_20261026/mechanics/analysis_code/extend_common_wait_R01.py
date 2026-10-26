#!/usr/bin/env python3
"""Qualify the retained R01 recovery, reduce the matched cohort and select stations."""
from pathlib import Path
import csv,hashlib,json,math,subprocess
import numpy as np
from primary_reference import raw_blocks,analyze,interval,moments,write_csv
from reduce_new_md import linear_cos_integral
from complete_common_wait import independent_integral
R=Path(__file__).resolve().parents[1];ROOT=R.parents[1]
PRIOR=R/'analysis/a10_common_wait_20261020'
OUT=R/'analysis/structure_common_wait_20261022/R01_extension'
RAW=ROOT/'merged_workspaces/CuTa-results'
REMOTE='/home/user13/NguyenDucToan/data/CuTa-results/validation/paper3_a10_common_wait_20261020'
CONDITIONS=['Baseline','A10B05','A10B20','A25B10','A25B20','A50B20']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ssh(code):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','cluster','python3','-'],input=code,text=True,capture_output=True,timeout=60)
 assert p.returncode==0,p.stderr
 return json.loads(p.stdout)
def recover():
 run=RAW/'runs/S4_R01_recovery';log=run/'logs/lammps.log';launch=run/'metadata/launch.json'
 certified=json.loads((ROOT/'paper1/reports/run_selection_20260924.json').read_text())['post_freeze_sensitivity']
 assert sha(log)==certified['R01_source_log']['sha256'] and sha(launch)==certified['R01_source_launch']['sha256']
 m=json.loads((PRIOR/'series/S4_R01.json').read_text());l=json.loads(launch.read_text());assert l['structure_sha256']==m['structure_sha256']
 blocks=[(h,a) for h,a in raw_blocks(log) if np.ptp(a[:,h.index('v_tool_x')])>1]
 assert len(blocks)==2;h,a=blocks[0];hb,b=blocks[1];assert h==hb and b[0,0]==575000
 oldh,old=next((hh,aa) for hh,aa in raw_blocks(R.parent/'evidence/S4_R01/lammps.log') if np.ptp(aa[:,hh.index('v_tool_x')])>1)
 assert oldh==h
 common={int(row[0]):row for row in old if row[0]<575000}
 prefix=a[a[:,0]<575000]
 for row in prefix:assert np.array_equal(row,common[int(row[0])]),int(row[0])
 discarded=int(np.sum(a[:,0]>=575000));assert discarded==5
 values=np.vstack([prefix,b]);assert np.all(np.diff(values[:,0])>0) and values[-1,0]==1114413
 text=log.read_text();assert certified['R01_completion_marker'] in text
 assert not any(x.startswith('ERROR:') for x in text.splitlines())
 d={k:values[:,i] for i,k in enumerate(h)};t=d['Time'];assert np.max(abs(t-values[:,0]*.001))<1e-9 and t[0]==161.080
 phase=np.unwrap(d['v_phase']*2*np.pi);omega=2*np.pi*m['frequency_per_ps'];offset=float(np.median(phase-omega*t))
 assert np.max(abs(phase-(omega*t+offset)))<1e-5
 assert np.ptp(d['v_tool_z']-m['amplitude_A']*np.sin(phase))<1e-3
 travel=(d['v_tool_x'][0]-d['v_tool_x'])/10;assert np.max(abs(travel-.03*(t-161.080)))<1e-5
 out=dict(time_ps=t,travel_nm=travel,Ft_nN=d['v_fx']*1.602176634,Fz_nN=d['v_fz']*1.602176634,z_A=d['v_tool_z'],theta_rad=phase,vz_A_ps=omega*m['amplitude_A']*np.cos(phase))
 m.update(historical_case_id='S4_R01_recovery',source_case_id='S4_R01_recovery',source_log=str(log),log_sha256=sha(log),source_launch_sha256=sha(launch),phase_offset=offset,completion_marker=certified['R01_completion_marker'],launch_completed=False,operational_completion='Qualified completed continuation; initial interrupted launcher retained',translation_start_ps=161.080,wait_ps=157.080,join=dict(checkpoint_step=575000,discarded_obsolete_rows=discarded,prefix_last_step=int(prefix[-1,0]),suffix_first_step=int(b[0,0]),prefix_matches_original_samples=len(prefix),selection='Retained recovery log prefix<575000 and continuation>=575000; old mechanical history superseded only in current selection'))
 receipt=dict(status='RECOVERY_HISTORY_QUALIFIED',source_log_sha256=sha(log),source_launch_sha256=sha(launch),initial_launcher_completed=False,completed_continuation=True,matched_original_prefix_rows=len(prefix),obsolete_rows_removed=discarded,new_dynamics_steps=0,prior_source_sha256=json.loads((PRIOR/'series/S4_R01.json').read_text())['log_sha256'],physical_variables=l['forwarded_variables'])
 (OUT/'recovery_qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
 return out,m
def mechanics():
 dest=OUT/'mechanics';assert not dest.exists();(dest/'series').mkdir(parents=True)
 replacement=recover();rows=[];cycles=[];sources=[];selected={};windows=[];quad=[];unchanged=[]
 previous={x['case']:x for x in csv.DictReader((PRIOR/'run_metrics.csv').open())}
 for condition in CONDITIONS:
  for rep in [1,2,3]:
   stem={'A25B10':'S4','A50B20':'S6'}.get(condition,'P3rev_wait_'+('Base' if condition=='Baseline' else condition));name=f'{stem}_R{rep:02d}'
   if name=='S4_R01':d,m=replacement
   else:
    a=np.genfromtxt(PRIOR/'series'/(name+'.csv'),names=True,delimiter=',');d={k:a[k] for k in a.dtype.names};m=json.loads((PRIOR/'series'/(name+'.json')).read_text())
   row,cc=analyze(d,m);row.update(condition=condition,realization=rep);rows.append(row);cycles.extend(cc);sources.append(m);selected[condition,rep]=(d,m)
   if name!='S4_R01':
    deltas=[abs(v-float(previous[name][k])) for k,v in row.items() if isinstance(v,(int,float)) and k in previous[name]]
    assert max(deltas)<1e-10;unchanged.append(dict(case=name,max_difference=max(deltas)))
   write_csv(dest/'series'/(name+'.csv'),[dict(zip(d,v)) for v in zip(*d.values())]);m['samples_sha256']=sha(dest/'series'/(name+'.csv'));(dest/'series'/(name+'.json')).write_text(json.dumps(m,indent=2)+'\n')
   if m['amplitude_A']:
    om=2*np.pi*m['frequency_per_ps'];origin=-m['phase_offset']/om
    for lo_nm in [4,6,8]:
     tt,ff=interval(d['time_ps'],d['Fz_nN'],161.080+lo_nm/.03,161.080+16/.03)
     error=abs(-.1*m['amplitude_A']*om*(linear_cos_integral(tt,ff,om,origin)-independent_integral(tt,ff,om,origin))/(tt[-1]-tt[0]));assert error<1e-9;quad.append(error)
 bases={x['realization']:x for x in rows if x['condition']=='Baseline'}
 for row in rows:
  base=bases[row['realization']];row['paired_reduction_percent']=100*(1-row['Ft_mean_nN']/base['Ft_mean_nN']);row['paired_input_increase_percent']=100*(row['total_power_uW']/base['total_power_uW']-1)
 def stats(vals):return dict(mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)))
 summary={c:{k:stats([x[k] for x in rows if x['condition']==c]) for k in ['Ft_mean_nN','Fz_mean_nN','tensile_percent','Pz_uW','total_power_uW','paired_reduction_percent','paired_input_increase_percent','stroke_reduction_percent']} for c in CONDITIONS}
 contrasts=[]
 for rep in [1,2,3]:
  rr={x['condition']:x for x in rows if x['realization']==rep}
  for a,b in [('A10B05','A10B20'),('A25B10','A25B20'),('A25B10','A50B20')]:contrasts.append(dict(first=a,second=b,realization=rep,force_reduction_percent=100*(1-rr[b]['Ft_mean_nN']/rr[a]['Ft_mean_nN']),input_increase_percent=100*(rr[b]['total_power_uW']/rr[a]['total_power_uW']-1)))
 for c in CONDITIONS:
  for rep in [1,2,3]:
   d,m=selected[c,rep];base,bm=selected['Baseline',rep]
   for low in [4,6,8]:
    lo=161.080+low/.03;hi=161.080+16/.03;ft=moments(d['time_ps'],d['Ft_nN'],lo,hi)[0];bf=moments(base['time_ps'],base['Ft_nN'],lo,hi)[0];tt,ff=interval(d['time_ps'],d['Fz_nN'],lo,hi);om=2*np.pi*m['frequency_per_ps'];pz=-.1*m['amplitude_A']*om*linear_cos_integral(tt,ff,om,-m['phase_offset']/om)/(hi-lo) if om else 0
    windows.append(dict(condition=c,realization=rep,start_nm=low,Ft=ft,Red=100*(1-ft/bf),Pz=pz,Pin=.03*ft+pz,paired_input_increase_percent=100*((.03*ft+pz)/(.03*bf)-1)))
 for name,data in [('run_metrics',rows),('complete_cycle_work',cycles),('contrasts',contrasts),('window_runs',windows)]:write_csv(dest/(name+'.csv'),data)
 report=dict(status='COMMON_WAIT_18_MECHANICAL_ANALYSIS_VERIFIED',independent_preparations=3,trajectories=18,window_nm=[4,16],translation_start_ps=161.080,wait_ps=157.080,summary=summary,contrasts=contrasts,sources=sources,maximum_gaussian_error_uW=max(quad),unchanged_17_histories=unchanged,lower_force_pairs=sum(x['paired_reduction_percent']>0 for x in rows if x['condition']!='Baseline'),higher_input_pairs=sum(x['paired_input_increase_percent']>0 for x in rows if x['condition']!='Baseline'),source_replacement='Only A25B10 R01, retained S4_R01_recovery matched to its structural station; prior cohort preserved',source_code_sha256=sha(Path(__file__)))
 (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('MECHANICS_READY',report['lower_force_pairs'],report['higher_input_pairs'],flush=True)
def select():
 assert not (OUT/'sources.json').exists()
 metadata=OUT/'mechanics/series';reference=ROOT/'results/runs/relax_laminate_R01/metadata/relaxed_structure.data'
 remotedirs={'A10B05':REMOTE+'/runs/P3rev_wait_A10B05_R01','A10B20':REMOTE+'/continuations/P3rev_wait_A10B20_R01/dgx01_segment01'}
 inventory=ssh('from pathlib import Path\nimport json\npaths='+repr(remotedirs)+"\nprint(json.dumps({c:[str(p) for p in (Path(d)/'trajectories').glob('scratch.*.dump')] for c,d in paths.items()}))")
 entries=[]
 for condition in CONDITIONS:
  stem={'A25B10':'S4','A50B20':'S6'}.get(condition,'P3rev_wait_'+('Base' if condition=='Baseline' else condition));case=stem+'_R01';mp=metadata/(case+'.json');m=json.loads(mp.read_text())
  assert sha(reference)==m['structure_sha256']
  if condition in remotedirs:paths={int(Path(p).name.split('.')[1]):p for p in inventory[condition]};remote=True
  else:
   base=RAW/('runs/S4_R01_recovery' if condition=='A25B10' else '.priority-recovery-pvZ3Ul/S6_R01' if condition=='A50B20' else 'validation/paper3_revision_v1_20261001_v4/runs/'+case)
   paths={int(p.name.split('.')[1]):str(p) for p in (base/'trajectories').glob('scratch.*.dump*')};remote=False
  eligible=sorted(n for n in paths if 4<=.03*(n*.001-161.080)<=16);assert eligible,case;step=eligible[-1];src=paths[step]
  phase=((2*np.pi*m['frequency_per_ps']*step*.001+m['phase_offset']-np.pi/2)%(2*np.pi))/np.pi if m['amplitude_A'] else None
  entries.append(dict(condition=condition,case=case,source_case_id='S4_R01_recovery' if condition=='A25B10' else case,preparation=1,role='station',step=step,travel_nm=.03*(step*.001-161.080),phase_over_pi=phase,target_phase_pi=None,phase_error_rad=None,path=str(OUT/'coordinates'/case/(Path(src).name+'.zst')) if remote else src,remote_source=src if remote else None,metadata=str(mp),reference=str(reference),reference_sha256=sha(reference)))
 selection=dict(status='FROZEN_BEFORE_R01_STRUCTURAL_OUTCOMES',protocol_sha256=sha(OUT/'protocol.json'),expected_record_count=6,expected_station_count=6,entries=entries)
 (OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 for entry in entries:
  src=entry['remote_source']
  if src:
   before=ssh('from pathlib import Path\nimport hashlib,json\np=Path('+repr(src)+')\na=p.stat();h=hashlib.sha256(p.read_bytes()).hexdigest();b=p.stat();assert(a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns);print(json.dumps(dict(bytes=b.st_size,mtime_ns=b.st_mtime_ns,sha256=h)))')
   dest=Path(entry['path']);dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
   remote=subprocess.Popen(['ssh','-C','-o','BatchMode=yes','cluster','cat',src],stdout=subprocess.PIPE)
   h=hashlib.sha256();count=0
   with dest.open('wb') as f:
    z=subprocess.Popen(['/opt/homebrew/bin/zstd','-q','-T1','-3','-c'],stdin=subprocess.PIPE,stdout=f)
    for b in iter(lambda:remote.stdout.read(1048576),b''):h.update(b);count+=len(b);z.stdin.write(b)
    z.stdin.close();assert z.wait()==0 and remote.wait()==0
   assert h.hexdigest()==before['sha256'] and count==before['bytes']
   after=ssh('from pathlib import Path\nimport json\np=Path('+repr(src)+');s=p.stat();print(json.dumps(dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)))');assert all(after[k]==before[k] for k in after)
   entry.update(raw_sha256=h.hexdigest(),raw_bytes=count,remote_before=before,remote_after=after)
  entry['source_sha256']=sha(entry['path']);entry['source_bytes']=Path(entry['path']).stat().st_size
 selection.update(status='R01_SELECTED_SOURCES_HASH_VERIFIED',raw_changed=False)
 (OUT/'sources.json').write_text(json.dumps(selection,indent=2)+'\n');print('R01_SOURCES_READY',len(entries),flush=True)
if __name__=='__main__':mechanics();select()
