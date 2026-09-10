"""Independent analytic checks of signs, weighting, and finite-window integrals."""
import unittest
import numpy as np
from rebuild_analysis import analyze, moments, negative_duration, CONV

class AnalysisChecks(unittest.TestCase):
    def test_linear_moments(self):
        mean,sd=moments(np.array([0.,1.]),np.array([0.,2.]),0.,1.)
        self.assertAlmostEqual(mean,1.)
        self.assertAlmostEqual(sd,1/np.sqrt(3))

    def test_zero_crossing(self):
        self.assertAlmostEqual(negative_duration(np.array([0.,1.]),np.array([-1.,3.]),0.,1.),.25)
        self.assertAlmostEqual(negative_duration(np.array([0.,1.]),np.array([3.,-1.]),0.,1.),.25)

    def test_prescribed_damping_work_and_stroke_sign(self):
        # Test-only analytic signal: force on tool opposes vertical velocity.
        t=np.linspace(0.,700.,70001);f=.037;amp=2.;omega=2*np.pi*f
        vz=omega*amp*np.cos(omega*t)
        d=dict(time_ps=t,Ft_nN=10+2*np.cos(omega*t),Fz_nN=-3*vz)
        meta=dict(case='analytic_test',amplitude_A=amp,frequency_per_ps=f,phase_offset=0.,sample_interval_ps=.01)
        result,cycles=analyze(d,meta)
        expected=3*amp**2*omega*np.pi/CONV
        self.assertAlmostEqual(result['work_eV_cycle']/expected,1.,places=5)
        self.assertGreater(result['up_nN'],result['down_nN'])
        self.assertLess(result['stroke_reduction_percent'],0)
        self.assertGreater(result['Pz_uW'],0)
        self.assertAlmostEqual((result['down_nN']*result['down_duration_ps']+result['up_nN']*result['up_duration_ps'])/400,result['Ft_mean_nN'])
        self.assertTrue(all(c['start_ps']>=result['window_start_ps'] and c['end_ps']<=result['window_end_ps'] for c in cycles))

if __name__=='__main__':unittest.main()
