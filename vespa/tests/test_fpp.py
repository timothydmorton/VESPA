from __future__ import print_function, division

import os, os.path
from vespa.fpp import FPPCalculation

INI_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                'fpp.ini'))

def test_fpp(ini_file=INI_FILE):
    f = FPPCalculation.from_ini('fpp.ini', n=100)

