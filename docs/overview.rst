.. _overview:

Overview
=========

.. py:currentmodule:: vespa

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
:class:`vespa.stars.StarPopulation`, which is useful beyond false positive
calculations, such as for generating a hypothetical population of
binary companions for a given star in order to help quantify
completeness to stellar companions of an imaging survey.

Installation
-----------

To install, you can get the most recently released version from PyPI::

    pip install vespa [--user]

Or you can clone the repository::

    git clone https://github.com/timothydmorton/vespa.git
    cd vespa
    python setup.py install [--user]

The ``--user`` argument may be necessary if you don't have root privileges.


Basic Usage
-----------

The simplest way to run an FPP calculation straight out of the box is
as follows.

  1.  Make a text file containing the transit photometry in three
  columns: ``t_from_midtransit`` [days], ``flux`` [relative,
  where out-of-transit is normalized to unity], and ``flux_err``.
  The file should not have a header row (no titles); and can be either
  whitespace or comma-delimited (will be ingested by
  :func:`np.loadtxt`).  

  2. Make a ``star.ini`` file that contains the observed properties of the target star (photometric and/or spectroscopic, whatever is available):: 

	    #provide spectroscopic properties if available
            #Teff = 3503, 80  #value, uncertainty
            #feh = 0.09, 0.09
            #logg = 4.89, 0.1

	    #observed magnitudes of target star
	    # If uncertainty provided, will be used to fit StarModel
            J = 9.763, 0.03
            H = 9.135, 0.03
            K = 8.899, 0.02
            Kepler = 12.473

  3. Make a ``fpp.ini`` file containing the following information::

            name = k2oi #anything
            ra = 11:30:14.510 #can be decimal form too
            dec = +07:35:18.21

            period = 32.988 #days
            rprs = 0.0534   #Rp/Rstar
            photfile = lc_k2oi.csv #contains transit photometry

	    [constraints]
	    maxrad = 12  # aperture radius [arcsec] 
	    secthresh = 1e-4 # Maximum allowed depth of potential secondary eclipse 

  4. Run the following from the command line (from within the same folder that has ``star.ini`` and ``fpp.ini``)::

	 $  calcfpp -n 1000

This will take a few minutes the first time you run it (note the
default simulation size is ``n=20000``, which would take longer but be
more reliable), and will output the FPP to the command line, as well
as producing diagnostic plots and a ``results.txt`` file with the
quantitative summary of the calculation.   In addition, this
will produce a number of data files in the same directory as your
``fpp.ini`` file:

  * ``trsig.pkl``: the pickled :class:`vespa.TransitSignal` object.
  * ``starfield.h5``: the TRILEGAL field star simulation
  * ``starmodel.h5``: the :class:`isochrones.StarModel` fit
  * ``popset.h5``: the :class:`vespa.PopulationSet` object
    representing the model population simulations.

It will also generate the following diagnostic plots:

  *  ``trsig.png``: A plot of the transit signal
  * ``eb.png``, ``heb.png``, ``beb.png``, ``pl.png``: plots
    illustrating the likelihood of each model.
  *  ``FPPsummary.png``: A summary figure of the FPP results.
  *  Summary plots of the
     :class:`isochrones.StarModel` fits.

Once these files have been created, it is faster to re-run the
calculation again, even if you change the constraints.

