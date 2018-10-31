from __future__ import print_function, division

import os, os.path
import unittest
import tempfile

from vespa.fpp import FPPCalculation

import pkg_resources

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

class TestFPP(unittest.TestCase):
    name = 'kepler-22'
    ini_file = 'fpp.ini'
    n = 200
    recalc = True
    refit_trap = False
    cadence = 0.02 # should be what is in ini files

    def setUp(self):
        dirname = os.path.join(ROOT, self.name)
        self.f = FPPCalculation.from_ini(dirname, ini_file=self.ini_file,
                                    n=self.n, recalc=self.recalc,
                                    refit_trap=self.refit_trap)

    def test_cadence(self):
        for pop in self.f.modelnames:
            assert self.f[pop].cadence == self.cadence

    def test_fpp(self):
        fpp = self.f.FPP()
        assert fpp > 0

    def test_bootstrap(self):
        h, lines = self.f.bootstrap_FPP(N=3)
        for line in lines:
            print("line:"+line+'\n'+"linesplit:"+(line.split()[-1]))
            assert float(line.split()[-1]) > 0

class TestFPP_CC(TestFPP):
    ini_file = 'fpp_cc.ini'
    recalc= False

class TestFPP_CC2(TestFPP):
    ini_file = 'fpp_cc2.ini'
    recalc = False
    
# class TestFPP_cadence(TestFPP):
#     ini_file = 'fpp_cadence.ini'
#     cadence = 0.01 # should be same as in fpp_cadence.ini

#     def test_cadence(self):
#         for pop in self.f.modelnames:
#             assert self.f[pop].cadence == self.cadence
