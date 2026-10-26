#!/usr/bin/env python3
"""Freeze selections and receive only exact common-wait structural configurations."""
import csv,hashlib,json,math,os,subprocess,tarfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];ROOT=R.parents[1]
OUT=R/'analysis/structure_common_wait_20261022'
REMOTE='/home/user13/NguyenDucToan/data/CuTa-results/validation/paper3_a10_common_wait_20261020'
OLD=ROOT/'merged_workspaces/CuTa-results/runs'
CONTROLS=ROOT/'merged_workspaces/CuTa-results/validation/paper3_revision_v1_20261001_v4/runs'
CONDITIONS=['Baseline','A10B05','A10B20','A25B10','A25B20','A50B20']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ssh(script):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','cluster','python3','-'],input=script,text=True,capture_output=True,timeout=60)
 assert p.returncode==0,p.stderr
 return json.loads(p.stdout)
def main():
 assert Path('/Volumes/CuTaData').is_mount() and not (OUT/'sources.json').exists()
 remote=ssh("from pathlib import Path\nimport json\nr=Path("+repr(REMOTE)+")\nprint(json.dumps({p.name:[x.name for x in sorted((p/'trajectories').glob('scratch.*.dump'))] for p in (r/'runs').glob('P3rev_wait_*') if p.name.endswith(('R02','R03'))}))")
 cycles=list(csv.DictReader((R/'analysis/a10_common_wait_20261020/complete_cycle_work.csv').open()))
 entries=[];cases=[]
 for prep in [2,3]:
  for condition in CONDITIONS:
   case=(f'S4_R{prep:02d}' if condition=='A25B10' else f'S6_R{prep:02d}' if condition=='A50B20' else f'P3rev_wait_{"Base" if condition=="Baseline" else condition}_R{prep:02d}')
   mp=R/'analysis/a10_common_wait_20261020/series'/(case+'.json');m=json.loads(mp.read_text());start=m.get('translation_start_ps',161.080)
   assert abs(start-161.080)<1e-8
   if condition.startswith('A10'):paths={int(n.split('.')[1]):str(Path(REMOTE)/'runs'/case/'trajectories'/n) for n in remote[case]};remote_case=True
   else:
    base=OLD if condition in ['A25B10','A50B20'] else CONTROLS
    paths={int(p.name.split('.')[1]):str(p) for p in (base/case/'trajectories').glob('scratch.*.dump*')};remote_case=False
   available=np.array(sorted(paths));eligible=available[(.03*(available*.001-start)>=4)&(.03*(available*.001-start)<=16)]
   assert len(eligible)>0
   selected=[('station',int(eligible[-1]),None,None)]
   cycle=None
   if prep==2 and condition!='Baseline':
    candidates=[c for c in cycles if c['case']==case];cycle=min(candidates,key=lambda c:abs(.03*((float(c['start_ps'])+float(c['end_ps']))/2-start)-15))
    aa=float(cycle['start_ps']);period=float(cycle['end_ps'])-aa;used=[]
    for j in range(8):
     target=(aa+j*period/8)*1000;step=int(eligible[np.argmin(abs(eligible-target))]);assert step not in used;used.append(step)
     phase=(2*math.pi*m['frequency_per_ps']*step*.001+m['phase_offset']-math.pi/2)%(2*math.pi)
     error=math.atan2(math.sin(phase-j*math.pi/4),math.cos(phase-j*math.pi/4));assert abs(error)<math.pi/4
     selected.append(('phase',step,j/4,error))
   for role,step,target,error in selected:
    path=paths[step];local=OUT/'coordinates'/case/'trajectories'/(Path(path).name+('.zst' if remote_case else '')) if remote_case else Path(path)
    entries.append(dict(condition=condition,case=case,preparation=prep,role=role,step=step,travel_nm=.03*(step*.001-start),phase_over_pi=((2*math.pi*m['frequency_per_ps']*step*.001+m['phase_offset']-math.pi/2)%(2*math.pi))/math.pi if m['amplitude_A'] else None,target_phase_pi=target,phase_error_rad=error,path=str(local),remote_source=path if remote_case else None,metadata=str(mp),reference=str(ROOT/f'results/runs/relax_laminate_R{prep:02d}/metadata/relaxed_structure.data'),reference_sha256=m['structure_sha256']))
   cases.append(dict(case=case,condition=condition,preparation=prep,available_frames=len(available),eligible_frames=len(eligible),phase_cycle=cycle))
 selection={'status':'FROZEN_BEFORE_STRUCTURAL_OUTCOMES','protocol_sha256':sha(OUT/'protocol.json'),'cases':cases,'entries':entries}
 (OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 wanted=sorted({e['remote_source'] for e in entries if e['remote_source']})
 remotehash=ssh('from pathlib import Path\nimport hashlib,json\npaths='+repr(wanted)+'''\nout={}\nfor n in paths:\n p=Path(n);a=p.stat();h=hashlib.sha256(p.read_bytes()).hexdigest();b=p.stat();assert(a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns);out[n]={'bytes':b.st_size,'mtime_ns':b.st_mtime_ns,'sha256':h}\nprint(json.dumps(out))''')
 (OUT/'remote_selection_before.json').write_text(json.dumps(remotehash,indent=2)+'\n')
 names=[str(Path(p).relative_to(REMOTE)) for p in wanted]
 proc=subprocess.Popen(['ssh','-C','-o','BatchMode=yes','cluster','tar','-C',REMOTE,'-cf','-','--',*names],stdout=subprocess.PIPE)
 received={}
 with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
  for member in archive:
   path=str(Path(REMOTE)/member.name);assert path in remotehash and member.isfile()
   target=OUT/'coordinates'/Path(member.name).relative_to('runs');target=Path(str(target)+'.zst');target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
   src=archive.extractfile(member);h=hashlib.sha256();count=0
   with target.open('wb') as dst:
    zipproc=subprocess.Popen(['/opt/homebrew/bin/zstd','-q','-T1','-3','-c'],stdin=subprocess.PIPE,stdout=dst)
    for chunk in iter(lambda:src.read(1048576),b''):h.update(chunk);count+=len(chunk);zipproc.stdin.write(chunk)
    zipproc.stdin.close();assert zipproc.wait()==0
   assert count==remotehash[path]['bytes'] and h.hexdigest()==remotehash[path]['sha256']
   received[path]=dict(path=str(target),compressed_sha256=sha(target),raw_sha256=h.hexdigest(),raw_bytes=count,compressed_bytes=target.stat().st_size)
   print('received',target.parent.name,target.name,flush=True)
 assert proc.wait()==0 and len(received)==len(wanted)
 after=ssh('from pathlib import Path\nimport json\npaths='+repr(wanted)+"\nprint(json.dumps({n:{'bytes':Path(n).stat().st_size,'mtime_ns':Path(n).stat().st_mtime_ns} for n in paths}))")
 for p,record in after.items():assert all(record[k]==remotehash[p][k] for k in record)
 for e in entries:
  p=Path(e['path']);assert p.exists();e['source_sha256']=sha(p);e['source_bytes']=p.stat().st_size
  if e['remote_source']:e.update(raw_sha256=received[e['remote_source']]['raw_sha256'],raw_bytes=received[e['remote_source']]['raw_bytes'])
  assert sha(e['reference'])==e['reference_sha256']
 result=dict(status='SELECTED_SOURCES_RECEIVED_AND_HASH_VERIFIED',protocol_sha256=sha(OUT/'protocol.json'),cases=cases,entries=entries,remote_before=remotehash,remote_after=after,remote_received=received,raw_changed=False)
 (OUT/'sources.json').write_text(json.dumps(result,indent=2)+'\n');print('READY',len(entries),'records',len(wanted),'received frames',flush=True)
def verify_existing():
 assert not (OUT/'sources.json').exists()
 selection=json.loads((OUT/'selection.json').read_text());before=json.loads((OUT/'remote_selection_before.json').read_text());entries=selection['entries']
 after=ssh('from pathlib import Path\nimport json\npaths='+repr(list(before))+"\nprint(json.dumps({n:{'bytes':Path(n).stat().st_size,'mtime_ns':Path(n).stat().st_mtime_ns} for n in paths}))")
 received={}
 for e in entries:
  if e['remote_source']:
   src=e['remote_source'];p=OUT/'coordinates'/Path(src).relative_to(Path(REMOTE)/'runs');p=Path(str(p)+'.zst');e['path']=str(p)
   assert all(after[src][k]==before[src][k] for k in ['bytes','mtime_ns'])
   raw=subprocess.check_output(['/opt/homebrew/bin/zstd','-dc',str(p)]);rawsha=hashlib.sha256(raw).hexdigest();assert rawsha==before[src]['sha256'] and len(raw)==before[src]['bytes']
   e['raw_sha256']=rawsha;e['raw_bytes']=len(raw);received[src]=dict(path=str(p),compressed_sha256=sha(p),raw_sha256=rawsha,raw_bytes=len(raw),compressed_bytes=p.stat().st_size)
  p=Path(e['path']);e['source_sha256']=sha(p);e['source_bytes']=p.stat().st_size;assert sha(e['reference'])==e['reference_sha256']
 result=dict(status='SELECTED_SOURCES_RECEIVED_AND_HASH_VERIFIED',protocol_sha256=sha(OUT/'protocol.json'),cases=selection['cases'],entries=entries,remote_before=before,remote_after=after,remote_received=received,raw_changed=False,path_layout_correction='Original collector retained trajectories/ in target layout; metadata-only finalization reconciles exact received paths, preserving initial selection and failed diagnostic; no file transfer repeated or selection changed.')
 (OUT/'sources.json').write_text(json.dumps(result,indent=2)+'\n');print('READY',len(entries),'records',len(received),'remote frames')
if __name__=='__main__':
 import sys
 if '--verify-existing' in sys.argv:verify_existing()
 else:main()

