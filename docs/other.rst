.. _other:

Other Utilities
=============

.. py:currentmodule:: vespa

Here are documented (occasionally sparsely) a few other utilities
used in the ``vespa`` package.

Plotting
-------

.. automodule:: vespa.plotutils
  :members:

Stats
-------

.. automodule:: vespa.statutils
  :members:

Hashing
---------

In order to be able to compare population objects, it's useful to define
utility functions to hash ``ndarrays`` and ``DataFrames`` and to
combine hashes in a legit way.  This is generally useful and could be its
own mini-package, but for now it's stashed here.
		
.. automodule:: vespa.hashutils
  :members:
