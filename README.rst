VESPA
======
.. image:: https://zenodo.org/badge/6253/timothydmorton/VESPA.svg   
    :target: http://dx.doi.org/10.5281/zenodo.16467


Validation of Exoplanet Signals using a Probabilistic Algorithm--- calculating false positive probabilities for transit signals

For usage and more info, `check out the documentation <http://vespa.rtfd.org>`_.

[Note: be aware that the documentation is out of date (though not totally useless) and I have not yet updated it; please email me or raise an issue if you have problems.]

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

For best results, it is also recommended to have ``MultiNest`` and ``pymultinest`` installed.  Without this, ``emcee`` will be used for stellar modeling, but the ``MulitNest`` results are a bit more trustworthy given the often multi-modal nature of stellar model fitting.

Basic Usage
-----------

The simplest way to run an FPP calculation straight out of the box is
as follows.

1. Make a text file containing the transit photometry in three columns: ``t_from_midtransit`` [days], ``flux`` [relative, where out-of-transit is normalized to unity], and ``flux_err``.  The file should not have a header row (no titles); and can be either whitespace or comma-delimited (will be ingested by ``np.loadtxt``).  

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
            rprs = 0.0534   #Rp/Rstar best estimate
            photfile = lc_k2oi.csv #contains transit photometry

	    [constraints]
	    maxrad = 12  # aperture radius [arcsec] 
	    secthresh = 1e-4 # Maximum allowed depth of potential secondary eclipse 

4. Run the following from the command line (from within the same folder that has ``star.ini`` and ``fpp.ini``)::

	$  calcfpp 
	 
Or, if you put the files in a folder called ``mycandidate``, then you can run ``calcfpp mycandidate``::
	 
This will run the calculation for you, creating result files, diagnostic plots, etc.  
It should take 20-30 minutes.  If you want to do a shorter
version to test, you can try ``calcfpp -n 1000`` (the default is 20000).  The first
time you run it though, about half the time is doing the stellar modeling, so it will still
take a few minutes.


Attribution
-----------

If you use this code, please cite both the paper and the code.

Paper citation::

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

code::

	@MISC{2015ascl.soft03011M,
	   author = {{Morton}, T.~D.},
	    title = "{VESPA: False positive probabilities calculator}",
	howpublished = {Astrophysics Source Code Library},
	     year = 2015,
	    month = mar,
	archivePrefix = "ascl",
	   eprint = {1503.011},
	   adsurl = {http://adsabs.harvard.edu/abs/2015ascl.soft03011M},
	  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
	}
