__version__ = '0.6'

try:
    __VESPA_SETUP__
except NameError:
    __VESPA_SETUP__ = False

if not __VESPA_SETUP__:

    __all__ = ['FPPCalculation',
               'EclipsePopulation', 'EBPopulation',
               'HEBPopulation', 'BEBPopulation',
               'PlanetPopulation', 'PopulationSet',
               'StarPopulation', 'MultipleStarPopulation',
               'BGStarPopulation', 'BGStarPopulation_TRILEGAL',
               'BinaryPopulation', 'Simulated_BinaryPopulation',
               'Raghavan_BinaryPopulation', 'TriplePopulation',
               'MAInterpolationFunction']


    #StarPopulation & children
    from .stars.populations import StarPopulation
    from .stars.populations import MultipleStarPopulation
    from .stars.populations import BGStarPopulation, BGStarPopulation_TRILEGAL
    from .stars.populations import BinaryPopulation
    from .stars.populations import Simulated_BinaryPopulation
    from .stars.populations import Raghavan_BinaryPopulation
    from .stars.populations import TriplePopulation

    from .transit_basic import MAInterpolationFunction


    #EclipsePopulation & children
    from .populations import EclipsePopulation
    from .populations import EBPopulation, HEBPopulation, BEBPopulation
    from .populations import PlanetPopulation
    from .populations import PopulationSet

    #from .populations import calculate_eclipses
    from .transitsignal import TransitSignal

    from .fpp import FPPCalculation
