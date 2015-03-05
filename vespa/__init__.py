__version__ = '0.1.1'

try:
    __VESPA_SETUP__
except NameError:
    __VESPA_SETUP__ = False

if not __VESPA_SETUP__:
    pass
