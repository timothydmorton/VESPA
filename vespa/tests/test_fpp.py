from __future__ import print_function, division

import os, os.path
from vespa.fpp import FPPCalculation

import pkg_resources

INI_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                'fpp.ini'))

def test_fpp(ini_file=INI_FILE):
    f = FPPCalculation.from_ini(ini_file, n=100)
    f.FPP()


