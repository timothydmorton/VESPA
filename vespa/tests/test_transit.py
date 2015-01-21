from numpy.testing import assert_allclose, assert_array_less, assert_equal
import numpy as np
from vespa import transit_basic as tr

def test_mafn():
    mafn = tr.MAInterpolationFunction()

def test_eclipse():
    p0, b, aR = tr.eclipse_pars(10, 1.0, 0.5, 1.0, 0.5)
    ts, fs = tr.eclipse(p0, b, aR, P=10)
    
    dur, depth, slope = tr.eclipse_tt(p0, b, aR, P=10)
    assert_allclose(dur, 0.191708658273, 0.01)    
    assert_allclose(depth, 0.289214147953, 0.01)    
    assert_allclose(slope, 2.93380430212, 0.01)
