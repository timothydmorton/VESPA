vespa
=================================

``vespa`` is a Python package built to enable automated false positive 
analysis of transiting planet signals. It implements the latest version of the 
general procedure described in detail in `Morton (2012) <http://adsabs.harvard.edu/abs/2012ApJ...761....6M>`_.

Installation
------------

To install, you can get the most recently released version from PyPI::

    pip install vespa [--user]

Or you can clone from github::

    git clone https://github.com/timothydmorton/vespa.git
    cd vespa
    python setup.py install [--user]

The ``--user`` argument may be necessary if you don't have root privileges.

Overview
--------

A false positive probability calculation in ``vespa`` is built of two basic
components: a :class:`TransitSignal` and a :class:`PopulationSet`.  The 
:class:`TransitSignal` holds the data about the transit signal photometry,
and the :class:`PopulationSet` contains a set of simulated populations, both
false positive scenarios (by default, this will be 
:class:`EBPopulation`, :class:`HEBPopulation`, :class:`BEBPopulation`) and
a true transiting planet population (:class:`PlanetPopulation`).   

Each of these population objects derives from :class:`EclipsePopulation`,
which in turn derives from :class:`StarPopulation`.  A :class:`StarPopulation`
contains a simulated population of stars, and can be used 


API Documentation
-----------------

.. toctree::
   :maxdepth: 2


   api
