#!/usr/bin/env python3
"""Render the verified common-wait ensemble; retain historical sources separately."""
import argparse, csv, hashlib, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from primary_reference import snapshot, analyze, analyze_legacy, interval, moments, write_csv
from reduce_new_md import linear_cos_integral

R=Path(__file__).resolve().parents[1];ROOT=R.parents[1]
D=R/'data';G=R/'generated';F=R/'figures/n3';F.mkdir(exist_ok=True)
AOUT=R/'analysis/a10_common_wait_20261020'
CASES=[('Baseline','scratch_base',0.,0.),('A10B05','S1',1.,.5),('A10B20','S2',1.,2.),
       ('A25B10','S4',2.5,1.),('A25B20','S5',2.5,2.),('A50B20','S6',5.,2.)]
COLORS=['#333333','#1b7837','#2166ac','#9970ab','#d6604d','#762a83']
METRICS={'Ft_mean_nN':'Force','Fz_mean_nN':'Fz','tensile_percent':'Tensile','down_nN':'Down',
         'up_nN':'Up','stroke_reduction_percent':'Stroke','paired_reduction_percent':'Reduction',
         'work_eV_cycle':'Work','Pz_uW':'Pz','total_power_uW':'Power'}
PREFIX=[['Base'],['Sone','AtenBfive'],['Stwo','AtenBtwenty'],['Sfour','AtwentyfiveBten'],
        ['Sfive','Mid','AtwentyfiveBtwenty'],['Ssix','High','AfiftyBtwenty']]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pm(values):return f'${np.mean(values):.2f} \\pm {np.std(values,ddof=1):.2f}$'
def stats(values):return {'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1))}
def table(name,header,rows,width=r'\textwidth'):
    (G/(name+'.tex')).write_text(r'\begin{tabular*}{'+width+r'}{@{\extracolsep{\fill}}l'+
        'c'*(len(header)-1)+r'@{}}\toprule'+'\n'+' & '.join(header)+r'\\\midrule'+'\n'+
        '\n'.join(' & '.join(x)+r' \\' for x in rows)+'\n'+r'\bottomrule\end{tabular*}'+'\n')
def fig():return plt.subplots(figsize=(3.45,2.5),layout='constrained')
def save(f,name):
    f.savefig(F/(name+'.pdf'));f.savefig(F/(name+'.png'),dpi=300);plt.close(f)

def main():
    global AOUT
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,default=AOUT);args=ap.parse_args();AOUT=args.analysis.resolve()
    report=json.loads((AOUT/'report.json').read_text())
    assert report['status']=='COMMON_WAIT_18_MECHANICAL_ANALYSIS_VERIFIED'
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'pdf.fonttype':42,
                         'xtick.direction':'in','ytick.direction':'in'})
    runs=[];cycles=[];series={};by_case={};provenance=[]
    for condition,stem,A,beta in CASES:
        for rep in (1,2,3):
            name=f'{stem}_R{rep:02d}' if condition in ('A25B10','A50B20') else f'P3rev_wait_{"Base" if condition=="Baseline" else condition}_R{rep:02d}'
            arr=np.genfromtxt(AOUT/'series'/f'{name}.csv',delimiter=',',names=True)
            d={k:arr[k] for k in arr.dtype.names};m=json.loads((AOUT/'series'/f'{name}.json').read_text())
            row,cc=analyze(d,m);row.update(condition=condition,realization=rep)
            runs.append(row);cycles.extend(cc);series[f'{stem}_R{rep:02d}']=(d,m);by_case[name]=(d,m);provenance.append(m)
    bases={x['realization']:x for x in runs if x['condition']=='Baseline'}
    for row in runs:row['paired_reduction_percent']=100*(1-row['Ft_mean_nN']/bases[row['realization']]['Ft_mean_nN'])
    groups={c:[r for r in runs if r['condition']==c] for c,*_ in CASES}
    summary={c:{k:stats([r[k] for r in rows]) for k in METRICS} for c,rows in groups.items()}
    write_csv(D/'run_metrics.csv',runs);write_csv(D/'complete_cycle_work.csv',cycles)
    (D/'condition_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    (D/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for name,keys,headers in [
        ('force_rows',['Ft_mean_nN','Fz_mean_nN','tensile_percent','paired_reduction_percent'],
         [r'Case',r'$\overline F_t$ (nN)',r'$\overline F_z$ (nN)',r'Tensile duty (\%)',r'Paired reduction (\%)']),
        ('power_rows',['Pz_uW','total_power_uW'],[r'Case',r'$P_z$ ($\mu$W)',r'$\overline P_{\rm in}$ ($\mu$W)'])]:
        table(name,headers,[[c]+[pm([x[k] for x in groups[c]]) for k in keys] for c,*_ in CASES],
              r'\linewidth' if name=='power_rows' else r'\textwidth')
    table('stroke_rows',[r'Case',r'Downstroke (nN)',r'Upstroke (nN)',r'Attenuation (\%)',r'$W_z$ (eV/cycle)',r'Cycles (R1/R2/R3)'],
          [[c]+[pm([x[k] for x in groups[c]]) for k in ['down_nN','up_nN','stroke_reduction_percent','work_eV_cycle']]+
           ['/'.join(str(x['complete_cycles']) for x in groups[c])] for c,*_ in CASES[1:]])
    table('paired_power_rows',[r'Case',r'$R_1$ (\%)',r'$P_1$ ($\mu$W)',r'$R_2$ (\%)',r'$P_2$ ($\mu$W)',r'$R_3$ (\%)',r'$P_3$ ($\mu$W)'],
          [[c]+[f'{x[k]:.2f}' for x in groups[c] for k in ['paired_reduction_percent','total_power_uW']] for c,*_ in CASES])
    original=(G/'numbers.tex').read_text();macro={}
    for (c,stem,A,beta),prefixes in zip(CASES,PREFIX):
        for key,suffix in METRICS.items():
            vals=[x[key] for x in groups[c]];mean=np.mean(vals);sd=np.std(vals,ddof=1)
            for prefix in prefixes:
                macro[prefix+suffix]=f'{mean:.2f}';macro[prefix+suffix+'SD']=f'{sd:.2f}'
                macro[prefix+suffix+'PM']=f'\\ensuremath{{{mean:.2f} \\pm {sd:.2f}}}'
                for x,tag in zip(vals,['Rone','Rtwo','Rthree']):macro[prefix+suffix+tag]=f'{x:.2f}'
        if A:
            dm=np.mean([.03*(bases[x['realization']]['Ft_mean_nN']-x['Ft_mean_nN']) for x in groups[c]])
            ratio=summary[c]['Pz_uW']['mean']/dm
            for prefix in prefixes:
                macro[prefix+'CostRatio']=macro['CostRatio'+prefix]=f'{ratio:.1f}'
                macro[prefix+'CostRatioExact']=macro['CostRatio'+prefix+'Exact']=f'{ratio:.2f}'
                macro[prefix+'DeltaPx']=f'{dm:.2f}'
                macro[prefix+'NormPower']=f"{summary[c]['total_power_uW']['mean']/summary['Baseline']['total_power_uW']['mean']:.2f}"
    base=summary['Baseline']['Ft_mean_nN']['mean'];high=summary['A50B20']['Ft_mean_nN']['mean']
    macro.update(BasePlat=f'{.03*base:.2f}',HighPlat=f'{.03*high:.2f}',HighPlatSaving=f'{.03*(base-high):.2f}',
                 PowerSurgePercent=f"{100*(summary['A50B20']['total_power_uW']['mean']/summary['Baseline']['total_power_uW']['mean']-1):.0f}")
    cost=[float(macro[p[0]+'CostRatioExact']) for p in PREFIX[1:]]
    macro.update(CostRatioMin=f'{min(cost):.1f}',CostRatioMax=f'{max(cost):.1f}')
    lines=[];seen=set()
    for line in original.splitlines():
        m=re.match(r'\\newcommand\{\\(\w+)\}\{',line)
        if m and m[1] in macro:lines.append(f'\\newcommand{{\\{m[1]}}}{{{macro[m[1]]}}}');seen.add(m[1])
        else:lines.append(line)
    lines.extend(f'\\newcommand{{\\{k}}}{{{v}}}' for k,v in macro.items() if k not in seen)
    (G/'numbers.tex').write_text('\n'.join(lines)+'\n')
    # All preparation-level work, power and sampling manifests.
    work=[];manifest=[]
    for row in runs:
        c=row['case'];rep=row['realization']; d,m=by_case[c]; name=c.replace('_',r'\_')
        px=.03*row['Ft_mean_nN'];pz=row['Pz_uW'];pin=row['total_power_uW'];norm=pin/bases[rep]['total_power_uW']
        work.append(dict(case_code=row['condition']+f'-R{rep:02d}',condition=row['condition'],realization=f'R{rep}',Ft_mean_nN=row['Ft_mean_nN'],Px_uW=px,Pz_uW=pz,Pin_uW=pin,Win_eV=pin*400*6.241509126,norm_Pin=norm))
        manifest.append([row['condition'],f'R{rep}',f"{m['sample_interval_ps']:.3f}",f"{row['window_start_ps']:.3f}",f"{row['window_end_ps']:.3f}",str(row['complete_cycles'])])
    write_csv(D/'supp_work_power.csv',work)
    (G/'supp_work_power_rows.tex').write_text('\n'.join(' & '.join([x['case_code'].replace('_',r'\_'),x['condition'],x['realization'],f"{x['Ft_mean_nN']:.2f}"]+[f"{x[k]:.4f}" for k in ['Px_uW','Pz_uW','Pin_uW']]+[f"{x['Win_eV']:.1f}",f"{x['norm_Pin']:.4f}"])+r' \\' for x in work)+'\n'+r'\bottomrule'+'\n')
    (G/'run_manifest_rows.tex').write_text('\n'.join(' & '.join(x)+r' \\' for x in manifest)+'\n'+r'\bottomrule'+'\n')
    wr=[];ws=[];q=[]
    for c,stem,A,beta in CASES:
        for start in (4.,6.,8.):
            vals=[]
            for rep in (1,2,3):
                d,m=series[f'{stem}_R{rep:02d}'];b,bm=series[f'scratch_base_R{rep:02d}']
                lo=d['time_ps'][0]+start/.03;hi=d['time_ps'][0]+16/.03
                ft=moments(d['time_ps'],d['Ft_nN'],lo,hi)[0]
                bf=moments(b['time_ps'],b['Ft_nN'],b['time_ps'][0]+start/.03,b['time_ps'][0]+16/.03)[0]
                tt,ff=interval(d['time_ps'],d['Fz_nN'],lo,hi);om=2*np.pi*m['frequency_per_ps']
                pz=float(-.1*A*om*linear_cos_integral(tt,ff,om,-m['phase_offset']/om)/((16.-start)/.03)) if A else 0.
                v=dict(Ft=ft,Red=100*(1-ft/bf),Pz=pz,Pin=.03*ft+pz);vals.append(v)
                wr.append(dict(case=f'{stem}_R{rep:02d}',condition=c,realization=rep,window_start_nm=start,window_end_nm=16.,**v))
            ws.append(dict(condition=c,window_nm=f'{start:g}--16',**{k+'_'+s:stats([v[k] for v in vals])[s] for k in vals[0] for s in ['mean','sd']}))
        if A:
            for rep in (1,2,3):
                d,m=series[f'{stem}_R{rep:02d}'];lo=d['time_ps'][0]+4/.03;hi=lo+400
                tt,ff=interval(d['time_ps'],d['Fz_nN'],lo,hi);om=2*np.pi*m['frequency_per_ps'];y=ff*om*A*np.cos(om*tt+m['phase_offset'])
                trap=-.1*np.trapezoid(y,tt)/400;simp=-.1*simpson(y,x=tt)/400
                exact=-.1*A*om*linear_cos_integral(tt,ff,om,-m['phase_offset']/om)/400
                v=dict(condition=c,realization=rep,Pz_trap_uW=float(trap),Pz_simp_uW=float(simp),Pz_exact_linear_force_uW=float(exact),diff_pct=float(100*abs(simp-trap)/trap))
                for stride in (2,4):
                    ix=np.unique(np.r_[np.arange(0,len(tt),stride),len(tt)-1]);coarse=-.1*np.trapezoid(y[ix],tt[ix])/400
                    v[f'downsample_{stride}x_pct']=float(100*abs(coarse-trap)/trap)
                q.append(v)
    write_csv(D/'supp_window_runs.csv',wr);write_csv(D/'supp_window_sensitivity.csv',ws);write_csv(D/'supp_quadrature_downsampling.csv',q)
    (G/'supp_window_rows.tex').write_text('\n'.join(' & '.join([v['condition'],v['window_nm']]+[f"${v[k+'_mean']:.2f} \\pm {v[k+'_sd']:.2f}$" for k in ['Ft','Red','Pz','Pin']])+r' \\' for v in ws)+'\n'+r'\bottomrule'+'\n')
    (G/'supp_quadrature_rows.tex').write_text('\n'.join(' & '.join([v['condition'],str(v['realization'])]+[f'{v[k]:.4f}' for k in ['Pz_trap_uW','Pz_simp_uW','diff_pct','downsample_2x_pct','downsample_4x_pct']])+r' \\' for v in q)+'\n'+r'\bottomrule'+'\n')
    profiles=[];phase=np.linspace(0,2*np.pi,129);items=[]
    for c,stem,A,beta in CASES[1:]:
        means={k:[] for k in ['Ft_nN','Fz_nN']};spreads={k:[] for k in means}
        for rep in (1,2,3):
            name=f'{stem}_R{rep:02d}';d,m=series[name];cc=[x for x in cycles if x['case']==m['case']]
            for k in means:
                a=np.array([np.interp(x['start_ps']+phase/(2*np.pi*m['frequency_per_ps']),d['time_ps'],d[k]) for x in cc])
                means[k].append(a.mean(axis=0));spreads[k].append(a.std(axis=0,ddof=1))
            profiles.extend(dict(condition=c,realization=rep,phi_rad=float(p),Ft_nN=float(ft),Fz_nN=float(fz),Ft_cycle_SD_nN=float(st),Fz_cycle_SD_nN=float(sz)) for p,ft,fz,st,sz in zip(phase,means['Ft_nN'][-1],means['Fz_nN'][-1],spreads['Ft_nN'][-1],spreads['Fz_nN'][-1]))
        items.append((c,A,{k:np.mean(v,axis=0) for k,v in means.items()},{k:np.sqrt(np.mean(np.array(v)**2,axis=0)) for k,v in spreads.items()}))
    write_csv(D/'phase_profiles.csv',profiles)
    for i,key in enumerate(['Ft_nN','Fz_nN','loop','difference']):
        f,ax=fig()
        for (c,A,mean,spread),color in zip(items,COLORS[1:]):
            if key in mean:
                ax.plot(phase/np.pi,mean[key],color=color,label=c);ax.fill_between(phase/np.pi,mean[key]-spread[key],mean[key]+spread[key],color=color,alpha=.12)
            elif key=='loop':
                xx=A*np.cos(phase); yy=mean['Fz_nN']
                ax.plot(xx,yy,color=color,label=c)
                for k in (28,92):
                    ax.annotate('',xy=(xx[k+5],yy[k+5]),xytext=(xx[k],yy[k]),arrowprops=dict(arrowstyle='->',color=color,lw=.8))
            else:ax.plot(10-A*np.cos(phase[:65]),mean['Ft_nN'][:65]-mean['Ft_nN'][::-1][:65],color=color,label=c)
        ax.set_xlabel(r'Phase $\phi/\pi$' if i<2 else ('Upward displacement (Å)' if i==2 else 'Nominal penetration (Å)'))
        ax.set_ylabel(['Lateral force (nN)','Normal force (nN)','Normal force (nN)','Downstroke − upstroke force (nN)'][i])
        if i<2:
            ax.axvline(1,color='.5',ls='--',lw=.6)
            ax.text(.25,.98,'Downstroke',transform=ax.transAxes,ha='center',va='top',fontsize=6.5)
            ax.text(.75,.98,'Upstroke',transform=ax.transAxes,ha='center',va='top',fontsize=6.5)
            ax.margins(y=.15)
        if i==0:ax.legend(ncol=2)
        save(f,'phase_response_'+chr(97+i))
    # Paired points for all preparations, mean and descriptive sample SD.
    f,ax=fig();ax.axhline(1,color='.5',ls='--',lw=.7)
    for (c,*_),color in zip(CASES,COLORS):
        x=np.array([r['paired_reduction_percent'] for r in groups[c]])
        y=np.array([r['total_power_uW']/bases[r['realization']]['total_power_uW'] for r in groups[c]])
        ax.plot(x,y,color=color,lw=.7,alpha=.5)
        for j,marker in enumerate(['o','s','^']):ax.scatter(x[j],y[j],c=color,marker=marker,s=17)
        ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),fmt='D',color=color,ms=4,capsize=2,label=c)
    from matplotlib.lines import Line2D
    condition_legend=ax.legend(ncol=2,loc='upper left',fontsize=6)
    ax.add_artist(condition_legend)
    handles=[Line2D([],[],color='.3',marker=m,ls='',markersize=4,label=lab) for m,lab in zip(['o','s','^','D'],['R01','R02','R03','Condition mean'])]
    ax.legend(handles=handles,loc='lower right',fontsize=5.7,ncol=2)
    ax.text(.68,.86,'Error bars: sample SD\n(n = 3)',transform=ax.transAxes,fontsize=5.8,ha='center')
    ax.set_xlabel('Paired force reduction (%)');ax.set_ylabel('Input / matched baseline power');save(f,'force_power_a')
    f,ax=fig();x=np.arange(5);saving=[[.03*(bases[r['realization']]['Ft_mean_nN']-r['Ft_mean_nN']) for r in groups[c]] for c,*_ in CASES[1:]]
    vertical=[[r['Pz_uW'] for r in groups[c]] for c,*_ in CASES[1:]]
    for offset,vals,color,label in [(-.18,saving,'#2e7d32','Lateral saving'),(.18,vertical,'#d35400','Normal input')]:
        ax.bar(x+offset,[np.mean(v) for v in vals],.36,yerr=[np.std(v,ddof=1) for v in vals],capsize=2,color=color,label=label)
    ax.text(.02,.98,'Error bars: sample SD (n = 3)',transform=ax.transAxes,va='top',fontsize=5.8)
    ax.set_xticks(x,[c[0] for c in CASES[1:]],rotation=25,ha='right');ax.set_ylabel('Mechanical power (µW)');ax.legend();save(f,'force_power_b')
    baseline=[]
    for key,letter in [('Ft_nN','a'),('Fz_nN','b')]:
        f,ax=fig()
        for rep,marker in zip((1,2,3),('o','s','^')):
            d,m=series[f'scratch_base_R{rep:02d}'];centres=[];values=[]
            for start in np.arange(4,16,2):
                end=min(start+2,16.);avg,sd=moments(d['time_ps'],d[key],d['time_ps'][0]+start/.03,d['time_ps'][0]+end/.03)
                centres.append((start+end)/2);values.append(avg)
                baseline.append(dict(realization=rep,quantity=key,start_nm=start,end_nm=end,mean_nN=avg,temporal_sd_nN=sd))
            ax.plot(centres,values,marker=marker,ms=3,label=f'R{rep}')
        ax.axvspan(4,16,color='.9');ax.set_xlabel('Tool travel (nm)');ax.set_ylabel(('Lateral' if key=='Ft_nN' else 'Normal')+' force (nN)');ax.legend();save(f,'baseline_blocks_'+letter)
    write_csv(D/'baseline_blocks.csv',baseline)
    # Existing main and SI tables now use all eighteen controlled histories.
    common=[]
    for row in runs:
        base=bases[row['realization']]
        common.append(dict(condition=row['condition'],preparation=row['realization'],mean_Ft_nN=row['Ft_mean_nN'],vertical_power_uW=row['Pz_uW'],net_input_power_uW=row['total_power_uW'],paired_reduction_percent=row['paired_reduction_percent'],paired_input_increase_percent=100*(row['total_power_uW']/base['total_power_uW']-1)))
    from complete_common_wait import table as ct
    keys=['mean_Ft_nN','vertical_power_uW','net_input_power_uW','paired_reduction_percent','paired_input_increase_percent']
    ct(G/'common_wait_rows.tex',['Condition',r'$\overline F_t$ (nN)',r'$P_z$ ($\mu$W)',r'$\overline P_{\rm in}$ ($\mu$W)',r'\shortstack{Paired force\\reduction (\%)}',r'\shortstack{Paired input\\increase (\%)}'],[[c]+[pm([x[k] for x in common if x['condition']==c]) for k in keys] for c,*_ in CASES])
    ct(G/'common_wait_individual_rows.tex',['Condition','Preparation',r'$\overline F_t$ (nN)',r'$P_z$ ($\mu$W)',r'$\overline P_{\rm in}$ ($\mu$W)',r'Reduction (\%)',r'Input increase (\%)'],[[x['condition'],str(x['preparation'])]+[f'{x[k]:.2f}' for k in keys] for x in common])
    table('baseline_rows',['Preparation',r'Window (nm)',r'$\overline F_t$ (nN)',r'Temporal SD (nN)',r'$\overline F_z$ (nN)',r'Temporal SD (nN)'],[[f'R{rep:02d}','4--16']+[f'{bases[rep][k]:.2f}' for k in ['Ft_mean_nN','Ft_temporal_SD_nN','Fz_mean_nN','Fz_temporal_SD_nN']] for rep in (1,2,3)])
    (AOUT/'render_receipt.json').write_text(json.dumps(dict(status='COMMON_WAIT_MECHANICAL_ASSETS_RENDERED',source_report_sha256=digest(AOUT/'report.json'),script_sha256=digest(Path(__file__)),primary_cases=[x['case'] for x in runs],figures=[f'figures/n3/{name}.pdf' for name in ['phase_response_a','phase_response_b','phase_response_c','phase_response_d','force_power_a','force_power_b','baseline_blocks_a','baseline_blocks_b']],notes=['Historical unequal-wait sources remain in original data records and private pre-edit checkpoint.','No manuscript mutation or simulation.']),indent=2)+'\n')
    print('Rendered common-wait main/SI numerical assets and eight existing figure panels.')

if __name__=='__main__':main()
