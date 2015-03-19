vespa
=================================

``vespa`` is a Python package built to enable automated false positive
probability (FPP) analysis of transiting planet signals. It implements
the latest version of the general procedure described in detail in
`Morton (2012) <http://adsabs.harvard.edu/abs/2012ApJ...761....6M>`_.

Installation
------------

To install, you can get the most recently released version from PyPI::

    pip install vespa [--user]

Or you can clone from github::

    git clone https://github.com/timothydmorton/vespa.git
    cd vespa
    python setup.py install [--user]

The ``--user`` argument may be necessary if you don't have root privileges.

Basic Usage
-----------

The simplest way to run an FPP calculation straight out of the box is
as follows.

  * Make a text file containing the transit photometry in three
  columns: ``t_from_midtransit`` [days], ``flux`` [relative,
  where out-of-transit is normalized to unity], and ``flux_err``.
  The file should not have a header row (no titles); and can be either
  whitespace or comma-delimited (will be ingested by
  :func:`np.loadtxt`).

  * Make a config file of the following form and save as ``fpp.ini``::

            name = k2oi #anything
            ra = 11:30:14.510 #can be decimal form too
            dec = +07:35:18.21

            period = 32.988 #days
            rprs = 0.0534   #Rp/Rstar
            photfile = lc_k2oi.csv #contains transit photometry

	    #provide Teff, feh, [logg optional] if spectrum available
            #Teff = 3503, 80  #value, uncertainty
            #feh = 0.09, 0.09
            #logg = 4.89, 0.1

	    #observed magnitudes of target star
	    # If uncertainty provided, will be used to fit StarModel
            [mags]
            J = 9.763, 0.03
            H = 9.135, 0.03
            K = 8.899, 0.02
            Kepler = 12.473

  * Run the following from the command line::

	 %  calcfpp -n 1000

This will take a few minutes the first time you run it (though the
default simulation size is ``n=20000``, which would take longer).


Overview
--------

A false positive probability calculation in ``vespa`` is built of two
basic components: a :class:`TransitSignal` and a
:class:`PopulationSet`, joined together in a :class:`FPPCalculation`
object.  The :class:`TransitSignal` holds the data about the transit
signal photometry, and the :class:`PopulationSet` contains a set of
simulated populations, one :class:`EclipsePopulation` for each
astrophysical model that is considered as a possible origin for the
observed transit-like signal.  By default, the populations included
will be :class:`PlanetPopulation` and three astrophysical false
positive scenarios: an :class:`EBPopulation`, an
:class:`HEBPopulation`, and a :class:`BEBPopulation`.

The :class:`EclipsePopulation` object derives from the more general
:class:`StarPopulation`, which is useful beyond false positive
calculations, such as for generating a hypothetical population of
binary companions for a given star in order to help quantify
completeness to stellar companions of an imaging survey.  



API Documentation
-----------------

.. toctree::
   :maxdepth: 2
	fpp
	api
