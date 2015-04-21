from __future__ import print_function, division

from vespa.populations import HEBPopulation
from vespa.populations import EBPopulation
from vespa.populations import BEBPopulation
from vespa.populations import PlanetPopulation

from isochrones.starmodel import StarModel
from isochrones.starmodel import BinaryStarModel
from isochrones.starmodel import TripleStarModel

from pkg_resources import resource_filename

import os, os.path
import tempfile
TMP = tempfile.gettempdir()

from vespa.transit_basic import MAInterpolationFunction

MAfn = MAInterpolationFunction(nzs=100,nps=200,pmin=0.007,pmax=1/0.007)

def test_heb(filename=os.path.join(TMP,'test_heb.h5')):
    mags = {'H': 10.211,
            'J': 10.523,
            'K': 10.152000000000001}
    period = 289.8622
    starmodel_file = resource_filename('vespa','tests/test_starmodel_triple.h5')
    starmodel = TripleStarModel.load_hdf(starmodel_file)
    pop = HEBPopulation(mags=mags, starmodel=starmodel,
                        period=period, n=100, MAfn=MAfn)
    pop.save_hdf(filename, overwrite=True)
    pop2 = HEBPopulation.load_hdf(filename)
    assert type(pop2)==HEBPopulation
    os.remove(filename)


def test_eb(filename=os.path.join(TMP,'test_eb.h5')):
    mags = {'H': 10.211,
            'J': 10.523,
            'K': 10.152000000000001}
    period = 289.8622
    starmodel_file = resource_filename('vespa','tests/test_starmodel_binary.h5')
    starmodel = BinaryStarModel.load_hdf(starmodel_file)
    pop = EBPopulation(mags=mags, starmodel=starmodel,
                       period=period, n=100, MAfn=MAfn)

    pop.save_hdf(filename, overwrite=True)
    pop2 = EBPopulation.load_hdf(filename)
    assert type(pop2)==EBPopulation
    os.remove(filename)

def test_beb(filename=os.path.join(TMP,'test_beb.h5')):
    trilegal_filename = resource_filename('vespa','data/kep22field.h5')
    mags = {'H': 10.211,
            'J': 10.523,
            'K': 10.152000000000001,
            'Kepler':12.0}
    period = 289.8622
    pop = BEBPopulation(period=period, mags=mags,
                        trilegal_filename=trilegal_filename,
                        n=100, MAfn=MAfn)

    pop.save_hdf(filename, overwrite=True)
    pop2 = BEBPopulation.load_hdf(filename)
    assert type(pop2)==BEBPopulation
    os.remove(filename)

def test_pl(filename=os.path.join(TMP,'test_pl.h5')):
    mass = (0.83,0.03)
    radius = (0.91,0.03)
    period = 289.8622
    rprs = 0.02
    starmodel_file = resource_filename('vespa','tests/test_starmodel_single.h5')
    starmodel = StarModel.load_hdf(starmodel_file)
    pop = PlanetPopulation(period=period, rprs=rprs,
                       starmodel=starmodel, n=100, MAfn=MAfn)

    pop.save_hdf(filename, overwrite=True)
    pop2 = PlanetPopulation.load_hdf(filename)
    assert type(pop2)==PlanetPopulation
    os.remove(filename)
