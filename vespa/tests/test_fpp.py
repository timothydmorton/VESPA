from __future__ import print_function, division

import os, os.path
from vespa.fpp import FPPCalculation

import pkg_resources

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def test_fpp(dirname='kepler-22'):
    dirname = os.path.join(ROOT,dirname)
    f = FPPCalculation.from_ini(dirname, n=100, recalc=True)
    f.FPP()

def test_fpp_cc(dirname='kepler-22'):
    dirname = os.path.join(ROOT, dirname)
    f = FPPCalculation.from_ini(dirname, ini_file='fpp_cc.ini', n=100)
    f.FPP()

def test_fpp_cc2(dirname='kepler-22'):
    dirname = os.path.join(ROOT, dirname)
    f = FPPCalculation.from_ini(dirname, ini_file='fpp_cc2.ini', n=100)
    f.FPP()
