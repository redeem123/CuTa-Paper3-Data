#!/usr/bin/env python3
"""Publish common-wait structural tables/plots after full numerical qualification."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];OUT=R/'analysis/structure_common_wait_20261022';TABLE=R/'generated/common_wait_structure';FIG=R/'figures/common_wait_structure_20261022'
CONDITIONS=['Baseline','A10B05','A10B20','A25B10','A25B20','A50B20'];COLORS=['#666666','#0072b2','#009e73','#9970ab','#e69f00','#762a83']
def csvwrite(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(name,header,rows,columns):
 s='\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}'+columns+'@{}}\\toprule\n'+header+r'\\\midrule'+'\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n'+r'\bottomrule\end{tabular*}'+'\n';(TABLE/(name+'.tex')).write_text(s)
def main():
 global OUT
 ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,default=OUT);ap.add_argument('--tables-only',action='store_true');args=ap.parse_args();OUT=args.evidence.resolve()
 d=json.loads((OUT/'report.json').read_text());assert d['status']=='COMMON_WAIT_STRUCTURE_AND_CONTACT_QUALIFIED';records=d['records'];stations=[e for e in records if e['role']=='station'];phases=[e for e in records if e['role']=='phase'];preparations=sorted({e['preparation'] for e in stations});assert len(stations)==6*len(preparations) and len(phases)==40
 parity=[]
 for prep in [2,3]:
  old=json.loads((R/'analysis'/('atomic_matched_review.json' if prep==2 else 'atomic_matched_review_R03.json')).read_text())
  for e in stations:
   if e['preparation']!=prep or e['condition'] not in ['A25B10','A50B20']:continue
   o=next(x for x in old['structural_rows'] if x['case']==e['case']);diff={k:abs(v-e['structure'][k]) for k,v in o.items() if k in e['structure'] and isinstance(v,(int,float))};assert max(diff.values())<1e-8
   oc=next(x for x in old['matched_contact'] if x['case']==e['case']);assert np.max(np.abs(np.array(oc['force_nN'])-e['contact']['force_nN']))<1e-10 and oc['repulsive_bulk_atoms']==e['contact']['repulsive_bulk_atoms']
   parity.append(dict(case=e['case'],max_existing_structural_difference=max(diff.values()),bulk_contacts_unchanged=True))
 oldphase=json.loads((R/'analysis/contact_phase_review.json').read_text());phaseparity=[]
 for e in phases:
  if e['condition'] not in ['A25B10','A50B20']:continue
  o=next(x for x in oldphase['rows'] if x['case']==e['case'] and x['step']==e['step']);assert o['repulsive_bulk_atoms']==e['contact']['repulsive_bulk_atoms'];err=abs(o['Fz_nN']-e['contact']['force_nN'][2]);assert err<1e-10;phaseparity.append(err)
 TABLE.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
 csvwrite(OUT/'station_structure.csv',[e['structure'] for e in stations])
 contactrows=[]
 for e in stations:
  c=e['contact'];contactrows.append(dict(condition=e['condition'],preparation=e['preparation'],case=e['case'],step=e['step'],travel_nm=e['travel_nm'],phase_over_pi=e['phase_over_pi'],cutoff_atoms=c['cutoff_metal_atoms'],repulsive_bulk_atoms=c['repulsive_bulk_atoms'],repulsive_nonbulk_atoms=c['repulsive_nonbulk_atoms'],Fx_nN=c['force_nN'][0],Fy_nN=c['force_nN'][1],Fz_nN=c['force_nN'][2]))
 csvwrite(OUT/'station_contact.csv',contactrows)
 phase_rows=[];summary=[]
 for e in phases:
  c=e['contact'];phase_rows.append(dict(condition=e['condition'],case=e['case'],step=e['step'],travel_nm=e['travel_nm'],target_phase_pi=e['target_phase_pi'],actual_phase_pi=e['phase_over_pi'],phase_error_rad=e['phase_error_rad'],cutoff_atoms=c['cutoff_metal_atoms'],repulsive_bulk_atoms=c['repulsive_bulk_atoms'],repulsive_nonbulk_atoms=c['repulsive_nonbulk_atoms'],Fx_nN=c['force_nN'][0],Fz_nN=c['force_nN'][2],repulsive_Fz_nN=c['repulsive_force_nN'][2],attractive_Fz_nN=c['attractive_force_nN'][2]))
 csvwrite(OUT/'phase_contact.csv',phase_rows)
 for condition in CONDITIONS[1:]:
  es=[e for e in phases if e['condition']==condition];negative=[e for e in es if e['contact']['force_nN'][2]<0]
  summary.append(dict(condition=condition,frames=len(es),travel_min_nm=min(e['travel_nm'] for e in es),travel_max_nm=max(e['travel_nm'] for e in es),bulk_min=min(e['contact']['repulsive_bulk_atoms'] for e in es),bulk_max=max(e['contact']['repulsive_bulk_atoms'] for e in es),nonbulk_min=min(e['contact']['repulsive_nonbulk_atoms'] for e in es),nonbulk_max=max(e['contact']['repulsive_nonbulk_atoms'] for e in es),tensile_frames=len(negative),tensile_with_bulk_contact=sum(e['contact']['repulsive_bulk_atoms']>0 for e in negative)))
 csvwrite(OUT/'phase_summary.csv',summary)
 for prep in preparations:
  ss=sorted([e for e in stations if e['preparation']==prep],key=lambda e:CONDITIONS.index(e['condition']));rows=[];packing=[];contacts=[];combined=[]
  for e in ss:
   s=e['structure'];c=e['contact'];rows.append([e['condition'],f"{e['travel_nm']:.3f}",'--' if e['phase_over_pi'] is None else f"{e['phase_over_pi']:.3f}",*[f"{s[k]:.2f}" for k in ['groove_deficit_nm3','pileup_nm3']],f"{s['median_shear']:.3f}",f"{s['median_D2min_A2']:.2f}",f"{s['threshold_volume_proxy_nm3']:.2f}"])
   packing.append([e['condition'],*[f"{s[k]:.2f}" for k in ['shear_above_0p15_percent','shear_above_0p30_percent','shear_above_0p50_percent']],f"{s['Cu_coordination_current']:.3f}",f"{s['Ta_coordination_current']:.3f}"])
   contacts.append([e['condition'],'--' if e['phase_over_pi'] is None else f"{e['phase_over_pi']:.3f}",str(c['cutoff_metal_atoms']),str(c['repulsive_bulk_atoms']),str(c['repulsive_nonbulk_atoms']),f"{c['force_nN'][0]:.2f}",f"{c['force_nN'][2]:.2f}"])
   combined.append(packing[-1]+[str(c['repulsive_bulk_atoms']),f"{c['force_nN'][2]:.2f}"])
  table(f'structure_R{prep:02d}',r'Condition & $\ell$ (nm) & $\phi/\pi$ & $V_g$ (nm$^3$) & $V_p$ (nm$^3$) & Median shear & Median $D^2_{\min}$ (\AA$^2$) & $V_{0.30}$ (nm$^3$)',rows,'lrrrrrrr')
  table(f'packing_R{prep:02d}',r'Condition & Shear $>0.15$ (\%) & $>0.30$ (\%) & $>0.50$ (\%) & Cu coordination & Ta coordination',packing,'lrrrrr')
  table(f'contact_R{prep:02d}',r'Condition & $\phi/\pi$ & Cutoff atoms & Repulsive bulk & Repulsive detached & $F_x$ (nN) & $F_z$ (nN)',contacts,'lrrrrrr')
  table(f'packing_contact_R{prep:02d}',r'Condition & $p_{0.15}$ & $p_{0.30}$ & $p_{0.50}$ & Cu coord. & Ta coord. & $N_{\rm rep,bulk}$ & $F_z$',combined,'lrrrrrrr')
 table('phase_summary',r'Condition & Travel range (nm) & Bulk-contact range & Detached-contact range & Tensile frames & Tensile with contact',[[s['condition'],f"{s['travel_min_nm']:.3f}--{s['travel_max_nm']:.3f}",f"{s['bulk_min']}--{s['bulk_max']}",f"{s['nonbulk_min']}--{s['nonbulk_max']}",str(s['tensile_frames']),str(s['tensile_with_bulk_contact'])] for s in summary],'lrrrrr')
 if not args.tables_only:
  plots(records,stations,phases)
 qual=dict(status='ALL_STATIONS_AND_40_PHASES_VERIFIED',station_count=len(stations),phase_count=len(phases),independent_station_preparations=len(preparations),phase_independent_preparations=1,station_travel_range_nm=[min(e['travel_nm'] for e in stations),max(e['travel_nm'] for e in stations)],max_phase_error_rad=max(abs(e['phase_error_rad']) for e in phases),max_engine_force_residual_nN=max(e['engine_validation']['max_abs_difference_nN'] for e in stations),max_manual_D2min_residual_A2=max(e['manual_strain_check']['max_D2min_difference_A2'] for e in stations),max_manual_shear_residual=max(e['manual_strain_check']['max_shear_difference'] for e in stations),all_ROI_fits_valid=all(e['structure']['invalid_roi_atoms']==0 for e in stations),all_surface_bins_valid=all(e['structure']['valid_surface_bins']==180 for e in stations),old_pair_station_parity=parity,old_pair_phase_max_force_error_nN=max(phaseparity),phase_summary=summary,connectivity_sensitivity=[dict(case=e['case'],default_bulk=e['contact']['repulsive_bulk_atoms'],sensitivity=e['connectivity_sensitivity']) for e in stations])
 (OUT/'qualification.json').write_text(json.dumps(qual,indent=2)+'\n');print(json.dumps({k:v for k,v in qual.items() if k not in ['connectivity_sensitivity','phase_summary']},indent=2))
def plots(records,stations,phases):
 plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
 fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained',sharex=True)
 for condition,color in zip(CONDITIONS[1:],COLORS[1:]):
  es=sorted([e for e in phases if e['condition']==condition],key=lambda e:e['phase_over_pi']);x=[e['phase_over_pi'] for e in es]
  axes[0].plot(x,[e['contact']['repulsive_bulk_atoms'] for e in es],'o-',ms=3,label=condition,color=color)
  axes[1].plot(x,[e['contact']['force_nN'][2] for e in es],'o-',ms=3,color=color)
 axes[0].set_ylabel('Repulsive bulk-contact atoms');axes[0].legend(ncol=3,fontsize=7);axes[1].set_ylabel('Coordinate reaction $F_z$ (nN)');axes[1].axhline(0,color='.5',lw=.7);axes[0].set_xlabel(r'Actual phase $\phi/\pi$');axes[1].set_xlabel(r'Actual phase $\phi/\pi$');axes[1].set_xlim(0,2);fig.savefig(FIG/'contact_phase.pdf');plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(7.1,3.1),layout='constrained',sharex=True,sharey=True)
 for ax,e in zip(axes.flat,[e for e in stations if e['preparation']==2]):
  z=np.load(e['fields']);im=ax.imshow(z['change'].T,origin='lower',extent=[28,46,2.8,12.8],cmap='coolwarm',vmin=-10,vmax=10,aspect='equal');ax.set_title(e['condition'],fontsize=9);ax.set_xlabel('$x$ (nm)');ax.set_ylabel('$y$ (nm)')
 fig.colorbar(im,ax=axes.ravel().tolist(),label='Surface height change (Å)',shrink=.8);fig.savefig(FIG/'height_maps.pdf');plt.close(fig)
if __name__=='__main__':main()
