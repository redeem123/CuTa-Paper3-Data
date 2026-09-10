"""Regression gates for scalar identities and figure provenance."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import unittest

ROOT=Path(__file__).resolve().parent

def rows(name):
    with (ROOT/'data'/name).open() as f:
        return list(csv.DictReader(f))

class ReproducibilityTests(unittest.TestCase):
    def test_all_power_identities(self):
        metrics={r['case']:r for r in rows('run_metrics.csv')}
        for r in rows('supp_work_power.csv'):
            case=r['case_code'];p=float(r['Pin_uW']);base=metrics['scratch_base_R0'+r['realization'][1]]
            self.assertAlmostEqual(float(r['Px_uW']),.03*float(metrics[case]['Ft_mean_nN']),places=12)
            self.assertAlmostEqual(p,float(r['Px_uW'])+float(r['Pz_uW']),places=12)
            self.assertAlmostEqual(float(r['Win_eV']),2496.6036504*p,places=8)
            self.assertAlmostEqual(float(r['norm_Pin']),p/float(base['total_power_uW']),places=12)
    def test_common_window_summary(self):
        metrics=rows('run_metrics.csv')
        for r in rows('supp_window_sensitivity.csv'):
            if r['window_nm']!='4.0--16.0':continue
            values=[float(m['total_power_uW']) for m in metrics if m['condition']==r['condition']]
            self.assertAlmostEqual(float(r['Pin_mean']),statistics.mean(values),places=12)
            self.assertAlmostEqual(float(r['Pin_std']),statistics.stdev(values),places=12)
    def test_visualization_sources(self):
        manifests=json.loads((ROOT/'data/visualization_manifest.json').read_text())
        self.assertEqual(len(manifests),6)
        for m in manifests.values():
            for k in ['reference','endpoint']:
                p=ROOT/m[k];self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),m[k+'_sha256'])
            self.assertTrue(m['atom_ids_match']);self.assertTrue(m['types_match'])
            self.assertEqual(m['atoms_workpiece'],543778)
            self.assertGreater(m['travel_nm'],16)
            self.assertNotEqual(m['endpoint_step'],13088)
            self.assertEqual(m['morphology_camera']['roi_x_A'],[180.,468.])
            self.assertEqual(m['morphology_camera']['fov_A'],140.)
            self.assertEqual(m['morphology_camera']['color'],'neutral gray; all species identical')
            for key in ['morphology_output','strain_output']:
                self.assertTrue((ROOT/m[key]).is_file())
    def test_retired_claims_absent(self):
        text='\n'.join((ROOT/n).read_text() for n in ['main_paper3.tex','supplementary.tex'])
        for phrase in ['was removed via boundary','negligible lateral clearance','substantial clearance','invariant to analysis window bounds','hold without discrepancy','phi_0 \\approx \\pi/2']:
            self.assertNotIn(phrase,text)

if __name__=='__main__':unittest.main()
