from __future__ import print_function, division

from ..populations import Raghavan_BinaryPopulation
from ..populations import MultipleStarPopulation
from ..populations import BGStarPopulation_TRILEGAL

import os, os.path
import tempfile
TMP = tempfile.gettempdir()

def test_raghavan(filename=os.path.join(TMP,'test_raghavan.h5')):
    pop = Raghavan_BinaryPopulation(1, n=100, qmin=0.2)
    pop.constrain_property('mass_B', lo=0.5)
    pop.save_hdf(filename, overwrite=True)
    pop2 = Raghavan_BinaryPopulation().load_hdf(filename)

    #test to make sure correct properties are there
    [pop2.q_fn, pop2.qmin, pop2.P_fn, pop2.ecc_fn, pop2.minmass]

    assert pop2.qmin == 0.2
    os.remove(filename)

def test_multiple(filename=os.path.join(TMP,'test_multiple.h5')):
    pop = MultipleStarPopulation(1, n=100, minmass=0.15)
    pop.constrain_property('mass_B',lo=0.5)
    pop.save_hdf(filename, overwrite=True)
    pop2 = MultipleStarPopulation().load_hdf(filename)

    #test to make sure correct properties are there
    [pop2.f_binary, pop2.f_triple, pop2.qmin, pop2.minmass,
     pop2.period_long_fn, pop2.period_short_fn, pop2.ecc_fn]

    assert pop2.minmass == 0.15

    os.remove(filename)

def test_multiple_specific_periods(filename=os.path.join(TMP,'test_pshort.h5')):
    pop = MultipleStarPopulation(1, period_short=100, n=100)
    pop.constrain_property('mass_B',lo=0.5)
    pop.save_hdf(filename, overwrite=True)
    pop2 = MultipleStarPopulation().load_hdf(filename)
    os.remove(filename)

    pop = MultipleStarPopulation(1, period_long=1000, n=100)
    pop.constrain_property('mass_B',lo=0.5)
    pop.save_hdf(filename, overwrite=True)
    pop2 = MultipleStarPopulation().load_hdf(filename)
    os.remove(filename)

#def test_bg(filename):
#    ra, dec = (289.21749900000003, 47.884459999999997) #Kepler-22
#    pop = BGStarPopulation_TRILEGAL('kepler22b.h5', ra, dec)
#    pop.save_hdf('test_bg.h5', overwrite=True)
#    pop2 = BGStarPopulation_TRILEGAL().load_hdf('test_bgpop.h5')
