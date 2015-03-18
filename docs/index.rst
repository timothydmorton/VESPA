vespa
=================================

``vespa`` is a Python package built to enable automated false positive 
analysis of transiting planet signals. It implements my latest version of the 
general procedure described in detail in `Morton (2012) <http://adsabs.harvard.edu/abs/2012ApJ...761....6M>`_.

Installation
------------

To install, you can get the most recently released version from PyPI::

    pip install isochrones [--user]

Or you can clone from github::

    git clone https://github.com/timothydmorton/isochrones.git
    cd isochrones
    python setup.py install [--user]

The ``--user`` argument may be necessary if you don't have root privileges.


API Documentation
-----------------

.. toctree::
   :maxdepth: 2


   api
