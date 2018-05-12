from setuptools import setup, Extension, find_packages

on_rtd = False
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    import numpy
except ImportError:
    on_rtd = True
    numpy = None
    build_ext = None

import os

def readme():
    with open('README.rst') as f:
        return f.read()

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__VESPA_SETUP__ = True
import vespa
version = vespa.__version__


# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

# Push a new tag to GitHub.
if "tag" in sys.argv:
    os.system("git tag -a {0} -m 'version {0}'".format(version))
    os.system("git push --tags")
    sys.exit()

if not on_rtd:
    transit_utils = [Extension('vespa._transitutils',['vespa/_transitutils.pyx'],
                                include_dirs=[numpy.get_include()])]
else:
    transit_utils = None

setup(name = "VESPA",
      version = version,
      description = "Calculate astrophysical false positive probabilities for transiting exoplanet signals",
      long_description = readme(),
      author = "Timothy D. Morton",
      author_email = "tim.morton@gmail.com",
      url = "https://github.com/timothydmorton/VESPA",
      #packages = ['vespa', 'vespa/stars',
      #            'vespa/orbits'],
      packages = find_packages(),
      package_data = {'vespa': ['data/*', 'tests/kepler-22/*.ini',
                                'tests/kepler-22/*.h5', 'tests/kepler-22/*.pkl',
                                'tests/kepler-22/*.cc', 'tests/kepler-22/signal.txt'],
                      'vespa.stars': ['data/*'],
                      'vespa.orbits':['data/*']},
      ext_modules = cythonize(transit_utils),
      scripts = ['scripts/get_trilegal',
                 'scripts/koifpp',
                 'scripts/batch_koifpp_condor',
                 'scripts/calcfpp',
                 'scripts/koifpp-config',
                 'scripts/get_kepler_ttvs'],
      cmdclass = {'build_ext': build_ext},
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy'
        ],
      install_requires=['pandas>=0.21','simpledist>=0.1.13', 'emcee',
                        'isochrones>=1.1.1', 'batman-package>=2.1',
                        'configobj'],
      zip_safe=False
)
