from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

def readme():
    with open('README.rst') as f:
        return f.read()

transit_utils = [Extension('transit_utils',['vespa/transit_utils.pyx'],
                                include_dirs=[numpy.get_include()])]

setup(name = "VESPA",
      version = "0.0",
      description = "Calculate astrophysical false positive probabilities for transiting exoplanet signals",
      long_description = readme(),
      author = "Timothy D. Morton",
      author_email = "tim.morton@gmail.com",
      url = "https://github.com/timothydmorton/VESPA",
      packages = find_packages(),
      package_data = {'vespa': ['data/*']},
      ext_modules = transit_utils,
      #scripts = ['scripts/write_cosi_dist',
      #           'scripts/calc_kappa_posterior'],
      cmdclass = {'build_ext': build_ext},
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy'
        ],
      install_requires=['cython','pandas>=0.13','simpledist','starutils','orbitutils'],
      zip_safe=False
) 
