#!/usr/bin/env python3
"""Render six final configurations with explicit ID and invalid-strain checks."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import AtomicStrainModifier, ColorCodingModifier
from ovito.vis import Viewport, TachyonRenderer
from ovito.pipeline import FileSource
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
CASES=[('base','scratch_base_R01'),('s1','S1_R01'),('s2','S2_R01'),('s4','S4_R01'),('s5','S5_R01'),('s6','S6_R01')]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def remove_tool(frame,data):
    data.particles_.delete_elements(np.asarray(data.particles['Particle Type'])==5)

def run(archive=False):
    out=ROOT/'evidence/visualization';out.mkdir(parents=True,exist_ok=True)
    source_root=Path('/Volumes/CuTaData/CuTa-results/runs')
    manifest_path=ROOT/'data/visualization_manifest.json'
    previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest={}
    for key,case in CASES:
        ref=out/f'{case}_reference.dump.gz';endpoint=out/f'{case}_final.dump.gz'
        if archive:
            launch=json.loads((ROOT/'evidence'/case/'launch.json').read_text())
            reference=Path(launch['structure'])
            if not reference.exists():
                reference=ROOT.parent/'results/runs/relax_laminate_R01/metadata/relaxed_structure.data'
            assert sha(reference)==launch['structure_sha256']
            final=source_root/case/'metadata/final.data'
            with final.open() as f: header=f.readline()
            step=int(re.search(r'timestep = (\d+)',header).group(1))
            p=import_file(str(final),atom_style='atomic');d=p.compute()
            rp=import_file(str(reference),atom_style='atomic');rd=rp.compute()
            # The scratch cell includes vacuum; do not create top-bottom neighbors in the reference.
            rd.cell_[...]=np.asarray(d.cell)
            export_file(rd,str(ref),'lammps/dump',columns=['Particle Identifier','Particle Type','Position.X','Position.Y','Position.Z'])
            pos=np.asarray(d.particles.positions);types=np.asarray(d.particles['Particle Type'])
            center=pos[types==5].mean(axis=0)
            p.modifiers.append(remove_tool)
            export_file(p,str(endpoint),'lammps/dump',columns=['Particle Identifier','Particle Type','Position.X','Position.Y','Position.Z'])
            m=dict(case=case, realization=1, source=str(final), source_sha256=sha(final),
                   reference_source=str(reference), reference_source_sha256=sha(reference),
                   reference_state='relaxed workpiece before scratch equilibration; not the last pre-translation frame',
                   endpoint_step=step, endpoint_time_ps=step*.001, tool_atom_centroid_A=center.tolist(),
                   reference=str(ref.relative_to(ROOT)), endpoint=str(endpoint.relative_to(ROOT)))
        else:
            m=previous[case].copy()
            assert sha(ref)==m['reference_sha256'] and sha(endpoint)==m['endpoint_sha256']
        import csv
        provenance={r['case']:r for r in csv.DictReader((ROOT/'data/run_provenance.csv').open())}
        start=float(provenance[case]['translation_start_ps'])
        m.update(translation_start_ps=start,travel_nm=.03*(m['endpoint_time_ps']-start),
                 reference_cell_note='Reference positions unchanged; cell includes scratch vacuum to avoid artificial top-bottom neighbors.')
        p=import_file(str(endpoint));r=import_file(str(ref)).compute();d=p.compute()
        ids=np.asarray(d.particles['Particle Identifier']);rid=np.asarray(r.particles['Particle Identifier'])
        assert len(ids)==len(set(ids))==543778
        assert np.array_equal(np.sort(ids),np.sort(rid))
        assert np.array_equal(np.asarray(d.particles['Particle Type'])[np.argsort(ids)],np.asarray(r.particles['Particle Type'])[np.argsort(rid)])
        # Atom IDs, not row order, define the reference correspondence.
        strain=AtomicStrainModifier(cutoff=3.375,affine_mapping=AtomicStrainModifier.AffineMapping.Off,select_invalid_particles=True)
        strain.reference=FileSource();strain.reference.load(str(ref));p.modifiers.append(strain)
        d=p.compute();shear=np.asarray(d.particles['Shear Strain']);invalid=np.asarray(d.particles['Selection']).astype(bool)|~np.isfinite(shear)
        m.update(reference_sha256=sha(ref),endpoint_sha256=sha(endpoint),atom_ids_match=True,types_match=True,
                 atoms_workpiece=len(ids),invalid_strain_count=int(invalid.sum()),
                 invalid_strain_policy='excluded from statistics; gray in strain rendering; never replaced by zero',
                 valid_shear_mean=float(shear[~invalid].mean()),valid_fraction_above_0_3=float(np.mean(shear[~invalid]>.3)),
                 cutoff_A=3.375,section_y_A=78.,section_thickness_A=30.,strain_range=[0.,1.],
                 morphology_output=f'figures/scratch_morph_{key}.png',strain_output=f'figures/scratch_strain_{key}.pdf')
        def chemistry(frame,data):
            pos=np.asarray(data.particles.positions)
            data.particles_.delete_elements((pos[:,0]<180)|(pos[:,0]>468)|(pos[:,2]<85))
            data.particles_.create_property('Color',data=np.tile([.68,.70,.72],(data.particles.count,1)))
            data.particles.vis.radius=1.35
        def hide_cell(frame,data):
            data.cell_.vis.enabled=False
        p.modifiers.append(hide_cell)
        p.modifiers.append(chemistry);p.add_to_scene()
        direction=np.array([-.45,.65,-.61]);direction/=np.linalg.norm(direction)
        target=np.array([324.,78.,111.])
        m['morphology_camera']=dict(projection='orthographic',direction=direction.tolist(),up=[0.,0.,1.],size=[1200,1000],target_A=target.tolist(),fov_A=140.,roi_x_A=[180.,468.],roi_z_min_A=85.,color='neutral gray; all species identical',shadows=True,ambient_occlusion=True)
        vp=Viewport(type=Viewport.Type.Ortho,camera_dir=tuple(direction),camera_up=(0,0,1),camera_pos=tuple(target-600*direction),fov=140.)
        raw_morph=out/f'{case}_morphology.png'
        vp.render_image(filename=str(raw_morph),size=(1200,1000),background=(1,1,1),renderer=TachyonRenderer(shadows=True,ambient_occlusion=True))
        fig,ax=plt.subplots(figsize=(6,5),dpi=200);fig.subplots_adjust(0,0,1,1)
        ax.imshow(plt.imread(raw_morph));ax.axis('off')
        right=np.cross(direction,[0.,0.,1.]);right/=np.linalg.norm(right)
        up=np.cross(right,direction)
        delta=np.array([-right[0],up[0]]);delta=delta/np.linalg.norm(delta)*190
        start=np.array([1030.,900.]);end=start+delta
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=2,color='black'))
        ax.text(980,955,'Scratch direction',ha='right',fontsize=13)
        ax.plot([80,80+50*1000/280],[920,920],color='black',lw=3)
        ax.text(80+25*1000/280,890,'5 nm',ha='center',fontsize=13)
        fig.savefig(ROOT/m['morphology_output'],dpi=200);plt.close(fig)
        p.modifiers.pop()
        p.modifiers.append(ColorCodingModifier(property='Shear Strain',start_value=0,end_value=1,gradient=ColorCodingModifier.Viridis()))
        def section(frame,data):
            invalid=np.asarray(data.particles['Selection']).astype(bool)|~np.isfinite(np.asarray(data.particles['Shear Strain']))
            data.particles_['Color_'][invalid]=[.65,.65,.65]
            data.particles_.delete_elements(np.abs(np.asarray(data.particles.positions)[:,1]-78)>15)
            data.particles.vis.radius=1.35
        p.modifiers.append(section)
        vp=Viewport(type=Viewport.Type.Ortho,camera_dir=(0,1,0),camera_pos=(234,-300,80),fov=80)
        raw=out/f'{case}_section.png'
        vp.render_image(filename=str(raw),size=(1800,600),background=(1,1,1),renderer=TachyonRenderer(shadows=False,ambient_occlusion=False))
        p.remove_from_scene()
        fig,ax=plt.subplots(figsize=(6,2),dpi=300);fig.subplots_adjust(0,0,1,1)
        ax.imshow(plt.imread(raw));ax.axis('off')
        # OVITO's orthographic fov is half the vertical span; the image is not cropped.
        pixels_per_A=600/160
        ax.plot([90,90+100*pixels_per_A],[540,540],color='white',lw=2)
        ax.text(90+50*pixels_per_A,520,'10 nm',ha='center',fontsize=9,color='white')
        ax.annotate('Scratch direction',xy=(90,70),xytext=(420,70),arrowprops=dict(arrowstyle='->'),fontsize=9,va='center')
        fig.savefig(ROOT/m['strain_output']);fig.savefig(ROOT/f'figures/scratch_strain_{key}.png',dpi=300);plt.close(fig)
        manifest[case]=m
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        print(case,'step',m['endpoint_step'],'invalid',m['invalid_strain_count'],'mean',m['valid_shear_mean'],flush=True)
    lines=[r'\begin{center}\small',r'\begin{tabular}{lrr}\toprule',r'Condition & Final step & Travel (nm)\\\midrule']
    for label,m in zip(['Baseline','A10B05','A10B20','A25B10','A25B20','A50B20'],manifest.values()):
        lines.append(f"{label} & {m['endpoint_step']} & {m['travel_nm']:.2f}"+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\end{center}']
    (ROOT/'generated/visualization_rows.tex').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(4.8,.5),layout='constrained')
    cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap='viridis'),cax=ax,orientation='horizontal')
    cb.set_label('Local shear strain; gray = unresolved',fontsize=9);cb.ax.tick_params(labelsize=8)
    fig.savefig(ROOT/'figures/scratch_strain_cbar.pdf');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',action='store_true');run(p.parse_args().archive)
