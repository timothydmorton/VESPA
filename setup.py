from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
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
      scripts = ['scripts/koifpp',
                 'scripts/batch_koifpp_condor'],
      cmdclass = {'build_ext': build_ext},
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy'
        ],
      install_requires=['cython','pandas>=0.13','simpledist>=0.1.11','starutils>=0.3','orbitutils>=0.1.5', 'emcee', 'hashutils>=0.0.3'],
      zip_safe=False
) 
