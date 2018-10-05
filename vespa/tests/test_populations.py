from __future__ import print_function, division

from pkg_resources import resource_filename

import os, os.path
import tempfile
import unittest
import logging

from pandas.util.testing import assert_frame_equal
import tables as tb

from vespa.populations import EclipsePopulation
from vespa.populations import HEBPopulation
from vespa.populations import EBPopulation
from vespa.populations import BEBPopulation
from vespa.populations import PlanetPopulation

from isochrones.starmodel import StarModel
from isochrones.starmodel import BinaryStarModel
from isochrones.starmodel import TripleStarModel

from vespa.transit_basic import MAInterpolationFunction

MAfn = MAInterpolationFunction(nzs=100,nps=200,pmin=0.007,pmax=1/0.007)

class MetaTestPopulation(type):
    @property
    def __test__(cls):
        return cls.population_type is not None

class TestPopulation(unittest.TestCase, metaclass=MetaTestPopulation):
    population_type = None
    starmodel_type = None
    models = 'mist'
    star_multiplicity = 'single'
    name = 'kepler-22'
    n = 100
    mags = {'H': 10.211, 'J': 10.523, 'K': 10.152, 'Kepler':11.664}
    period = 289.8622
    cadence = 0.02

    def setUp(self):
        self.pop = self.population_type(**self.population_kwargs)

    @property
    def population_kwargs(self):
        return self._get_population_kwargs()

    def _get_population_kwargs(self):
        filename = os.path.join('tests', self.name,
                                '{}_starmodel_{}.h5'.format(self.models, self.star_multiplicity))
        starmodel_file = resource_filename('vespa', filename)
        starmodel = self.starmodel_type.load_hdf(starmodel_file)
        return dict(mags=self.mags, starmodel=starmodel,
                    period=self.period, n=self.n, MAfn=MAfn, cadence=self.cadence)

    def test_roundtrip(self):
        with tempfile.NamedTemporaryFile() as file:
            self.pop.save_hdf(file.name, overwrite=True)
            assert len(tb.file._open_files.get_handlers_by_name(file.name)) == 0

            pop2 = self.population_type.load_hdf(file.name)
            assert len(tb.file._open_files.get_handlers_by_name(file.name)) == 0
            assert type(pop2) == self.population_type
            assert_frame_equal(self.pop.stars, pop2.stars)

            assert self.pop.cadence == pop2.cadence

            try:
                assert_frame_equal(self.pop.starmodel.samples, pop2.starmodel.samples)
                assert self.pop.starmodel.print_ascii() == pop2.starmodel.print_ascii()
            except AttributeError:
                pass

    def test_kwargs(self):
        assert self.period == self.pop.period
        assert self.n == self.pop.n
        assert self.cadence == self.pop.cadence

        if hasattr(self.pop, 'mags'):
            assert self.mags == self.pop.mags

    def test_resample(self):
        pop2 = self.pop.resample()
        assert len(pop2.stars) == len(self.pop.stars)

class TestHEBPopulation(TestPopulation):
    population_type = HEBPopulation
    starmodel_type = TripleStarModel
    star_multiplicity = 'triple'

class TestEBPopulation(TestPopulation):
    population_type = EBPopulation
    starmodel_type = BinaryStarModel
    star_multiplicity = 'binary'

class TestBEBPopulation(TestPopulation):
    population_type = BEBPopulation

    def _get_population_kwargs(self):
        filename = os.path.join('tests', self.name, 'starfield.h5')
        trilegal_filename = resource_filename('vespa', filename)
        return dict(mags=self.mags, trilegal_filename=trilegal_filename,
                    period=self.period, n=self.n, MAfn=MAfn, cadence=self.cadence)

class TestPlanetPopulation(TestPopulation):
    population_type = PlanetPopulation
    starmodel_type = StarModel
    star_multiplicity = 'single'
    rprs = 0.02

    def _get_population_kwargs(self):
        kwargs = super()._get_population_kwargs()
        del kwargs['mags']
        kwargs['rprs'] = self.rprs
        return kwargs
