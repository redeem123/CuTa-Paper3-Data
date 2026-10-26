#!/usr/bin/env python3
"""Portable common-wait structure/contact reduction of selected saved coordinates."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['QT_QPA_PLATFORM']='offscreen'
import argparse,csv,hashlib,json,subprocess,math,warnings
warnings.filterwarnings('ignore',message='.*OVITO.*PyPI')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial import cKDTree
from ovito.io import import_file,export_file
from ovito.modifiers import AtomicStrainModifier,CoordinationAnalysisModifier
from ovito.pipeline import FileSource
import ovito
import common_wait_structure_kernel as K
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frame(entry):
 p=Path(entry['path']);assert sha(p)==entry['source_sha256'],str(p)
 if p.suffix=='.zst':raw=subprocess.check_output(['zstd','-dc',str(p)])
 else:raw=p.read_bytes()
 h=hashlib.sha256(raw).hexdigest()
 if 'raw_sha256' in entry:assert h==entry['raw_sha256']
 lines=raw.split(b'\n',9);head=[x.decode().strip() for x in lines[:9]];fields=head[8].split()[2:]
 assert int(head[1])==entry['step'];values=np.fromstring(lines[9].decode(),sep=' ').reshape(-1,len(fields));assert len(values)==int(head[3])
 atoms=values[:,[fields.index(x) for x in ['id','type','x','y','z']]].copy();bounds=np.array([list(map(float,x.split()[:2])) for x in head[5:8]])
 assert len(np.unique(atoms[:,0]))==len(atoms) and np.isfinite(atoms).all()
 return atoms,bounds,h

def manual_check(ids,ref,cur,shear,d2,valid,Ly):
 xyz=ref+np.array([1000.,0.,1000.]);xyz[:,1]%=Ly;tree=cKDTree(xyz,boxsize=[1e6,Ly,1e6]);ii=np.flatnonzero(valid);indices=ii[::max(1,len(ii)//12)][:12];rows=[]
 for i in indices:
  js=np.array(tree.query_ball_point(xyz[i],3.375));js=js[js!=i];a=ref[js]-ref[i];b=cur[js]-cur[i];a[:,1]-=np.rint(a[:,1]/Ly)*Ly;b[:,1]-=np.rint(b[:,1]/Ly)*Ly
  J=np.linalg.lstsq(a,b,rcond=None)[0];res=b-a@J;nonaff=float(np.sum(res*res));eps=.5*(J@J.T-np.eye(3));dev=eps-np.trace(eps)/3*np.eye(3);gamma=float(np.sqrt(.5*np.sum(dev*dev)))
  rows.append(dict(atom_id=int(ids[i]),neighbours=len(js),D2min_manual_A2=nonaff,D2min_field_A2=float(d2[i]),shear_manual=gamma,shear_field=float(shear[i])))
 md=max(abs(v['D2min_manual_A2']-v['D2min_field_A2']) for v in rows);ms=max(abs(v['shear_manual']-v['shear_field']) for v in rows);assert md<1e-5 and ms<1e-6,(md,ms)
 return dict(max_D2min_difference_A2=md,max_shear_difference=ms,rows=rows)

def calculate(job):
 e,work,potential,engine,skip=job;directory=Path(work)/e['case']/(e['role']+'_'+str(e['step']));directory.mkdir(parents=True,exist_ok=True);K.CACHE=directory;K.POTENTIAL=Path(potential);K.ENGINE=Path(engine)
 target=directory/'result.json'
 if target.exists():
  result=json.loads(target.read_text());assert result['source_sha256']==e['source_sha256'] and result['kernel_sha256']==sha(Path(K.__file__))
  if e['role']=='station':assert Path(result['fields']).is_file()
  return result
 atoms,bounds,rawhash=frame(e);metal=atoms[atoms[:,1]!=5];metal=metal[np.argsort(metal[:,0])];bulk,graph=K.bulk_graph(metal,bounds);contact=K.pair_contact(atoms,bounds,bulk)
 result=dict(e);result.update(raw_sha256=rawhash,kernel_sha256=sha(Path(K.__file__)),contact=contact,connectivity=graph)
 if e['role']=='station':
  assert sha(e['reference'])==e['reference_sha256']
  reference=import_file(e['reference']);reference.modifiers.append(K.remove_tool);reference.modifiers.append(CoordinationAnalysisModifier(cutoff=3.375));rd=reference.compute()
  ids=np.asarray(rd.particles['Particle Identifier']);order=np.argsort(ids);ids=ids[order].copy();xyz=np.asarray(rd.particles['Position'])[order].copy();types=np.asarray(rd.particles['Particle Type'])[order].copy();refcoord=np.asarray(rd.particles['Coordination'])[order].copy()
  assert np.array_equal(metal[:,0],ids) and np.array_equal(metal[:,1],types)
  refdump=directory/'reference.dump';export_file(rd,str(refdump),'lammps/dump',columns=['Particle Identifier','Particle Type','Position.X','Position.Y','Position.Z'])
  current=directory/'metal.dump';K.write_dump(current,metal,bounds,e['step']);p=import_file(str(current));p.modifiers.append(K.remove_tool)
  strain=AtomicStrainModifier(cutoff=3.375,affine_mapping=AtomicStrainModifier.AffineMapping.Off,select_invalid_particles=True,output_nonaffine_squared_displacements=True);strain.reference=FileSource();strain.reference.load(str(refdump));p.modifiers.append(strain);p.modifiers.append(CoordinationAnalysisModifier(cutoff=3.375));d=p.compute();index=np.argsort(np.asarray(d.particles['Particle Identifier']))
  pos=np.asarray(d.particles['Position'])[index].copy();shear=np.asarray(d.particles['Shear Strain'])[index].copy();d2=np.asarray(d.particles['Nonaffine Squared Displacement'])[index].copy();coord=np.asarray(d.particles['Coordination'])[index].copy();invalid=np.asarray(d.particles['Selection'])[index].astype(bool)|~np.isfinite(shear)|~np.isfinite(d2)
  top=xyz[:,2].max();roi=(xyz[:,0]>=390)&(xyz[:,0]<440)&(xyz[:,1]>=33)&(xyz[:,1]<123)&(xyz[:,2]>=top-50)&(xyz[:,2]<top-5);valid=roi&~invalid
  assert valid.sum()==roi.sum()>0
  change=K.height(pos[bulk])-K.height(xyz);xx=np.arange(36)*5+282.5;yy=np.arange(20)*5+30.5;region=(xx[:,None]>=390)&(xx[:,None]<440)&(yy[None,:]>=33)&(yy[None,:]<123)&np.isfinite(change);assert region.sum()==180
  s=dict(condition=e['condition'],case=e['case'],preparation=e['preparation'],step=e['step'],travel_nm=e['travel_nm'],phase_over_pi=e['phase_over_pi'],tracked_metal_atoms=len(ids),reference_roi_atoms=int(roi.sum()),valid_roi_atoms=int(valid.sum()),invalid_roi_atoms=int((roi&invalid).sum()),invalid_strain_all=int(invalid.sum()),mean_shear=float(shear[valid].mean()),median_shear=float(np.median(shear[valid])),mean_D2min_A2=float(d2[valid].mean()),median_D2min_A2=float(np.median(d2[valid])),threshold_volume_proxy_nm3=float(202.5*np.mean(shear[valid]>.3)),groove_deficit_nm3=float(25*np.maximum(-change[region],0).sum()/1000),pileup_nm3=float(25*np.maximum(change[region],0).sum()/1000),valid_surface_bins=int(region.sum()))
  for q,label in [(.15,'0p15'),(.3,'0p30'),(.5,'0p50')]:s['shear_above_'+label+'_percent']=float(100*np.mean(shear[valid]>q))
  for species,label in [([1,3],'Cu'),([2,4],'Ta')]:
   mask=roi&np.isin(types,species);s[label+'_coordination_reference']=float(refcoord[mask].mean());s[label+'_coordination_current']=float(coord[mask].mean())
  result['structure']=s;result['manual_strain_check']=manual_check(ids,xyz,pos,shear,d2,valid,bounds[1,1]-bounds[1,0])
  result['engine_validation']={'status':'SKIPPED_EXPLICITLY','integration_steps':0} if skip else K.engine_check(e['case'],e['step'],atoms,bounds,contact)
  sensitivity=[]
  for cutoff in [3.2,3.6]:
   bm,bg=K.bulk_graph(metal,bounds,cutoff);cc=K.pair_contact(atoms,bounds,bm);sensitivity.append(dict(**bg,repulsive_bulk_atoms=cc['repulsive_bulk_atoms'],repulsive_nonbulk_atoms=cc['repulsive_nonbulk_atoms']))
  result['connectivity_sensitivity']=sensitivity
  dest=Path(work)/'fields';dest.mkdir(exist_ok=True);field=dest/(e['case']+'.npz');np.savez_compressed(field,ids=ids,types=types,reference=xyz,position=pos,shear=shear,nonaffine=d2,invalid=invalid,bulk=bulk,change=change,roi=roi)
  result['fields']=str(field);result['fields_sha256']=sha(field)
 target.write_text(json.dumps(result,indent=2)+'\n');return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--work',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--case');ap.add_argument('--potential',type=Path,default=R.parents[1]/'potentials/CuTa_LJ15_2014.adp.txt');ap.add_argument('--engine',type=Path,default=Path('/opt/homebrew/Caskroom/miniforge/base/envs/hea-md/bin/lmp'));ap.add_argument('--skip-engine',action='store_true');a=ap.parse_args();source=json.loads(a.manifest.read_text());entries=source['entries'];assert len(entries)==source.get('expected_record_count',52) and sum(e['role']=='station' for e in entries)==source.get('expected_station_count',12)
 assert len({(e['case'],e['role'],e['step']) for e in entries})==len(entries)
 assert a.skip_engine or(a.engine.is_file() and a.potential.is_file());a.work.mkdir(parents=True,exist_ok=True);results=[]
 if a.case:entries=[e for e in entries if e['case']==a.case]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(calculate,(e,str(a.work),str(a.potential),str(a.engine),a.skip_engine)) for e in entries]
  for f in as_completed(futures):
   result=f.result();results.append(result);c=result['contact'];print(result['case'],result['role'],result['step'],'bulk',c['repulsive_bulk_atoms'],'Fz',round(c['force_nN'][2],4),flush=True)
 order={c:i for i,c in enumerate(['Baseline','A10B05','A10B20','A25B10','A25B20','A50B20'])};results.sort(key=lambda e:(e['preparation'],order[e['condition']],e['role'],e['step']))
 report=dict(status='COMMON_WAIT_STRUCTURE_AND_CONTACT_QUALIFIED' if not a.skip_engine else 'STRUCTURE_REPLAY_ENGINE_CHECK_SKIPPED',ovito_version=ovito.version_string,source_manifest_sha256=sha(a.manifest),analysis_sha256=sha(Path(__file__)),kernel_sha256=sha(Path(K.__file__)),station_preparations=sorted({e['preparation'] for e in results if e['role']=='station'}),phase_preparation=sorted({e['preparation'] for e in results if e['role']=='phase'}),records=results)
 a.output.write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE',len(results),str(a.output),flush=True)
if __name__=='__main__':main()
