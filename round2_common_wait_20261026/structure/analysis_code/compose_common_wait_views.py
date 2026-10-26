#!/usr/bin/env python3
"""Compose all six common-wait R02 surface/shear views at fixed display scales."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
R=Path(__file__).resolve().parents[1];F=R/'figures/common_wait_structure_20261022'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest=json.loads((R/'analysis/structure_common_wait_20261022/render.json').read_text());items={e['condition']:e for e in manifest['views']};conditions=['Baseline','A10B05','A10B20','A25B10','A25B20','A50B20'];fig,axes=plt.subplots(4,3,figsize=(7.1,7.2));fig.subplots_adjust(left=.015,right=.985,top=.96,bottom=.08,hspace=.26,wspace=.045);inputs={}
 for mode,offset in [('surface',0),('strain',6)]:
  for i,c in enumerate(conditions):
   e=items[c];v=next(v for v in e['views'] if v['mode']==mode);pdf=R/v['output'];assert sha(pdf)==v['output_sha256'];png=F/(e['case']+'_'+mode+'.png');inputs[png.name]=sha(png);im=plt.imread(png);ax=axes.flat[offset+i];h,w=im.shape[:2];ax.imshow(im);ax.axis('off');ax.set_title(f'({chr(97+offset+i)}) {c}: '+('surface' if mode=='surface' else 'shear'),fontsize=8.2,pad=2)
   ppA=h/(2*v['fov_half_vertical_span_A']);left=100;y=h-60;ax.plot([left,left+20*ppA],[y,y],color='k',lw=1.2);ax.text(left+10*ppA,y-25,'2 nm',ha='center',fontsize=6.5,bbox=dict(facecolor='white',edgecolor='none',pad=.2,alpha=.9));ax.annotate('',xy=(w*.69,h*.12),xytext=(w*.93,h*.12),arrowprops=dict(arrowstyle='->',lw=.8));ax.text(w*.81,h*.055,'Scratch direction',ha='center',fontsize=6)
 cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0,1),cmap='viridis'),cax=fig.add_axes([.35,.025,.3,.014]),orientation='horizontal');cb.set_label('Local shear strain',fontsize=8,labelpad=2);cb.ax.tick_params(labelsize=7,length=2)
 dest=F/'atomic_views.pdf';fig.savefig(dest,dpi=300);plt.close(fig);(R/'analysis/structure_common_wait_20261022/composition.json').write_text(json.dumps(dict(output=str(dest.relative_to(R)),sha256=sha(dest),panels=12,conditions=conditions,inputs=inputs,source='Actual R02 common-wait coordinate fields; no synthetic atoms or field alteration',station_travel_range_nm=[15.8676,15.9807],section_thickness_A=6),indent=2)+'\n')
if __name__=='__main__':main()
