#!/usr/bin/env python3
"""Verify and independently reduce the supplied structural-data archive."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,math
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--engine',type=Path);a=ap.parse_args();w=a.work.resolve();assert not w.exists();w.mkdir(parents=True)
 files=json.loads((R/'MANIFEST.json').read_text())['files']
 for n,v in files.items():assert sha(R/n)==v['sha256'],n
 source=json.loads((R/'sources.json').read_text())
 for e in source['entries']:
  e['path']=str(R/e['path']);e['reference']=str(R/e['reference']);e['metadata']=str(R/e['metadata']);m=json.loads(Path(e['metadata']).read_text());expected=(2*math.pi*m['frequency_per_ps']*e['step']*.001+m['phase_offset']-math.pi/2)%(2*math.pi)/math.pi if m['amplitude_A'] else None
  if expected is not None:assert abs(expected-e['phase_over_pi'])<1e-12
  assert abs(e['travel_nm']-.03*(e['step']*.001-161.080))<1e-10
 manifest=w/'sources_resolved.json';manifest.write_text(json.dumps(source,indent=2)+'\n');output=w/'replayed.json'
 command=[sys.executable,str(R/'analysis_code/analyze_common_wait_structure.py'),'--manifest',str(manifest),'--work',str(w/'computation'),'--output',str(output),'--workers',str(a.workers),'--potential',str(R/'potential/CuTa_LJ15_2014.adp.txt')]
 if a.engine:command+=['--engine',str(a.engine.resolve())]
 else:command+=['--skip-engine']
 with (w/'replay.log').open('w') as log:p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,str(w/'replay.log')
 actual=json.loads(output.read_text());expected=json.loads((R/'results/report.json').read_text());lookup={(e['case'],e['role'],e['step']):e for e in expected['records']};count=0;error=0
 def compare(a,b):
  nonlocal count,error
  if isinstance(a,(float,int)) and not isinstance(a,bool):
   delta=abs(a-b);error=max(error,delta);assert delta<=1e-7,(a,b);count+=1
  elif isinstance(a,list):
   assert len(a)==len(b)
   for aa,bb in zip(a,b):compare(aa,bb)
  elif isinstance(a,dict):
   for k in a:
    if k in b:compare(a[k],b[k])
 for e in actual['records']:
  old=lookup[e['case'],e['role'],e['step']];compare(e['contact'],old['contact']);compare(e['connectivity'],old['connectivity'])
  if e['role']=='station':compare(e['structure'],old['structure']);compare(e['manual_strain_check'],old['manual_strain_check']);compare(e['connectivity_sensitivity'],old['connectivity_sensitivity'])
 stations=sum(e['role']=='station' for e in actual['records']);phases=sum(e['role']=='phase' for e in actual['records'])
 assert stations==source.get('expected_station_count',12) and stations+phases==source.get('expected_record_count',52)
 result=dict(status='STRUCTURAL_DATA_REPLAY_PASS',records=len(actual['records']),station_count=stations,phase_count=phases,numeric_values_compared=count,max_abs_numeric_difference=error,file_hashes_verified=len(files),coordinate_force_engine_repeated=bool(a.engine),supplied_original_zero_step_checks=f'{stations} passed checks are retained in results/qualification.json and verification_logs/',new_dynamics_steps=0)
 (w/'replay_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
