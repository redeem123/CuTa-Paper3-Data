#!/usr/bin/env python3
"""Replay separately retained interaction and heat-sink sensitivity histories."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'analysis_code'))
from primary_reference import analyze

def main():
 expected=list(csv.DictReader((R/'controls/control_metrics.csv').open()));values=[];count=0;error=0
 for row in expected:
  n=row['case'];p=R/'controls'/(n+'.csv');m=json.loads((R/'controls'/(n+'.json')).read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==m['samples_sha256']
  a=np.genfromtxt(p,delimiter=',',names=True);d={k:a[k] for k in a.dtype.names};current,_=analyze(d,m)
  for k,v in current.items():
   if k in row and isinstance(v,(int,float)):
    delta=abs(v-float(row[k]));assert delta<1e-8,(n,k,delta);count+=1;error=max(error,delta)
  values.append(dict(case=n,Ft_mean_nN=current['Ft_mean_nN'],tensile_percent=current['tensile_percent'],Pz_uW=current['Pz_uW'],total_power_uW=current['total_power_uW']))
 unit=[]
 for n in ['S5_R01','S5_R02']:
  p=R/'controls'/(n+'.csv');m=json.loads((R/'controls'/(n+'.json')).read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==m['samples_sha256'];a=np.genfromtxt(p,delimiter=',',names=True);v,_=analyze({k:a[k] for k in a.dtype.names},m);unit.append(dict(case=n,Ft_mean_nN=v['Ft_mean_nN'],tensile_percent=v['tensile_percent']))
 result=dict(status='SENSITIVITY_HISTORY_REPLAY_PASS',adopted_controls=5,unit_scale_comparison_histories=2,numeric_values_compared=count,max_abs_difference=error,unit_scale_force_mean_nN=float(np.mean([v['Ft_mean_nN'] for v in unit])),unit_scale_force_SD_nN=float(np.std([v['Ft_mean_nN'] for v in unit],ddof=1)),unit_scale_tensile_mean_percent=float(np.mean([v['tensile_percent'] for v in unit])),unit_scale_tensile_SD_percent=float(np.std([v['tensile_percent'] for v in unit],ddof=1)),unit_scale_histories=unit,values=values,scope='Separate original matched histories, never pooled into18 common-wait primary trajectories')
 (R/'sensitivity_replay_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
