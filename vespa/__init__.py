__version__ = '0.1.2'

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
               'ColormatchMultipleStarPopulation',
               'Spectroscopic_MultipleStarPopulation',
               'BGStarPopulation', 'BGStarPopulation_TRILEGAL',
               'BinaryPopulation',
               'MAInterpolationFunction']

    from .fpp import FPPCalculation

    #EclipsePopulation & children
    from .populations import EclipsePopulation
    from .populations import EBPopulation, HEBPopulation, BEBPopulation
    from .populations import PlanetPopulation
    from .populations import PopulationSet

    #from .populations import calculate_eclipses
    from .transitsignal import TransitSignal
    
    #StarPopulation & children
    from .stars.populations import StarPopulation
    from .stars.populations import MultipleStarPopulation
    from .stars.populations import ColormatchMultipleStarPopulation
    from .stars.populations import Spectroscopic_MultipleStarPopulation
    from .stars.populations import BGStarPopulation, BGStarPopulation_TRILEGAL
    from .stars.populations import BinaryPopulation

    from .transit_basic import MAInterpolationFunction

    pass
