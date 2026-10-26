#!/usr/bin/env python3
"""Common-wait atomistic views, unchanged camera and field definitions.
Renderer function copied from source SHA256: 5321fd4ce2c03ce40f35e71123ec470d553e0e9c2ddf767fa114460f1157f79c
"""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import warnings
warnings.filterwarnings('ignore',message='.*OVITO.*PyPI')
import hashlib,json,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from ovito.data import DataCollection,Particles
from ovito.pipeline import Pipeline,StaticSource
from ovito.vis import Viewport,TachyonRenderer
import ovito
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
R=Path(__file__).resolve().parents[1]
F=R/'figures/common_wait_structure_20261022'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def render(case,mode,z):
 pos=z['position'];invalid=z['invalid'];shear=z['shear']
 if mode=='surface':
  mask=(pos[:,0]>=280)&(pos[:,0]<=460)&(pos[:,1]>=28)&(pos[:,1]<=128)&(pos[:,2]>=85)
  colors=np.tile([.66,.69,.72],(int(mask.sum()),1));direction=np.array([0.,.6,-.8]);target=np.array([370.,78.,111.]);fov=55.;size=(2160,1200)
 else:
  mask=(pos[:,0]>=280)&(pos[:,0]<=460)&(np.abs(pos[:,1]-78)<=3)&(pos[:,2]>=60)&(pos[:,2]<=135)
  colors=plt.get_cmap('viridis')(Normalize(0,1,clip=True)(shear[mask]))[:,:3];colors[invalid[mask]]=[.65,.65,.65]
  direction=np.array([0.,1.,0.]);target=np.array([370.,78.,97.5]);fov=42.;size=(2160,960)
 data=DataCollection();particles=Particles();data.objects.append(particles)
 particles.create_property('Position',data=pos[mask]);particles.create_property('Particle Identifier',data=z['ids'][mask]);particles.create_property('Color',data=colors)
 particles.vis.radius=1.1
 pipeline=Pipeline(source=StaticSource(data=data));pipeline.add_to_scene()
 camera=Viewport(type=Viewport.Type.Ortho,camera_dir=tuple(direction),camera_up=(0,0,1),camera_pos=tuple(target-direction*600),fov=fov)
 raw=F/(case+'_'+mode+'.png')
 camera.render_image(filename=str(raw),size=size,background=(1,1,1),renderer=TachyonRenderer(shadows=False,ambient_occlusion=(mode=='surface'),antialiasing=True))
 pipeline.remove_from_scene()
 fig=plt.figure(figsize=(3.25,2.2));ax=fig.add_axes([.03,.07,.94,.90] if mode=='surface' else [.03,.07,.75,.90])
 ax.imshow(plt.imread(raw));ax.set_xlim(0,size[0]);ax.set_ylim(size[1],0);ax.axis('off')
 ppA=size[1]/(2*fov);left=110;bar_y=size[1]-65
 ax.plot([left,left+20*ppA],[bar_y,bar_y],color='black',lw=1.8)
 ax.text(left+10*ppA,bar_y-24,'2 nm',ha='center',fontsize=8,color='black',bbox=dict(facecolor='white',edgecolor='none',pad=.4,alpha=.9))
 ax.annotate('',xy=(size[0]*.62,150),xytext=(size[0]*.92,150),arrowprops=dict(arrowstyle='->',lw=1.2))
 ax.text(size[0]*.77,65,'Scratch direction',ha='center',va='center',fontsize=8)
 if mode=='strain':
  cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0,1),cmap='viridis'),cax=fig.add_axes([.81,.24,.025,.55]));cb.set_label('Local shear strain',fontsize=7.5,labelpad=3);cb.ax.tick_params(labelsize=7,length=2)
 fig.savefig(F/(case+'_'+mode+'.pdf'),dpi=600);plt.close(fig)
 return dict(mode=mode,atoms_displayed=int(mask.sum()),selection=('280<=x<=460;28<=y<=128;z>=85 A' if mode=='surface' else '280<=x<=460;75<=y<=81;60<=z<=135 A'),projection='orthographic',direction=direction.tolist(),target_A=target.tolist(),fov_half_vertical_span_A=fov,image_size=list(size),sphere_radius_A=1.1,colors=('neutral gray with ambient occlusion' if mode=='surface' else 'unchanged shear, viridis [0,1]; invalid gray'),scale_bar_nm=2,invalid_displayed=int(invalid[mask].sum()),output=str((F/(case+'_'+mode+'.pdf')).relative_to(R)),output_sha256=sha(F/(case+'_'+mode+'.pdf')))

def one(entry):
 F.mkdir(parents=True,exist_ok=True)
 with np.load(entry['fields']) as z:
  assert sha(Path(entry['fields']))==entry['fields_sha256']
  views=[render(entry['case'],mode,z) for mode in ['surface','strain']]
 return dict(case=entry['case'],condition=entry['condition'],views=views,source_sha256=entry['source_sha256'],fields_sha256=entry['fields_sha256'],travel_nm=entry['travel_nm'],phase_over_pi=entry['phase_over_pi'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();d=json.loads((R/'analysis/structure_common_wait_20261022/report.json').read_text());entries=[e for e in d['records'] if e['role']=='station' and e['preparation']==2];assert len(entries)==6
 results=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for f in as_completed([pool.submit(one,e) for e in entries]):
   v=f.result();results.append(v);print(v['condition'],'rendered',flush=True)
 (R/'analysis/structure_common_wait_20261022/render.json').write_text(json.dumps(dict(ovito_version=ovito.version_string,renderer='Tachyon',section_thickness_A=6,shear_scale=[0,1],views=results),indent=2)+'\n')
if __name__=='__main__':main()
