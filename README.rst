VESPA
======
.. image:: https://zenodo.org/badge/6253/timothydmorton/VESPA.svg   
    :target: http://dx.doi.org/10.5281/zenodo.16467


Validation of Exoplanet Signals using a Probabilistic Algorithm--- calculating false positive probabilities for transit signals

For usage and more info, `check out the documentation <http://vespa.rtfd.org>`_.

Installation
------------

To install, you can get the most recently released version from PyPI::

    pip install vespa [--user]

Or you can clone the repository::

    git clone https://github.com/timothydmorton/vespa.git
    cd vespa
    python setup.py install [--user]

The ``--user`` argument may be necessary if you don't have root privileges.

Depends on typical scientific packages (e.g. `numpy`, `scipy`, `pandas`),
as well as `isochrones <http://github.com/timothydmorton/isochrones>`_, and (in several corners of the code), another package of mine called `simpledist <http://github.com/timothydmorton/simpledist>`_.  All dependencies *should* get resolved upon install, though this has only been tested under the anaconda Python distribution, which has all the scientific stuff already well-organized.

Attribution
-----------

If you use this code, please cite the following paper::

    @ARTICLE{2012ApJ...761....6M,
    author = {{Morton}, T.~D.},
    title = "{An Efficient Automated Validation Procedure for Exoplanet Transit Candidates}",
    journal = {\apj},
    archivePrefix = "arXiv",
    eprint = {1206.1568},
    primaryClass = "astro-ph.EP",
    keywords = {planetary systems, stars: statistics },
    year = 2012,
    month = dec,
    volume = 761,
    eid = {6},
    pages = {6},
    doi = {10.1088/0004-637X/761/1/6},
    adsurl = {http://adsabs.harvard.edu/abs/2012ApJ...761....6M},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

