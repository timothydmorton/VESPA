from __future__ import print_function, division

import os, os.path
from vespa.fpp import FPPCalculation

import pkg_resources

FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def test_fpp(folder=FOLDER):
    f = FPPCalculation.from_ini(folder, n=100)
    f.FPP()


