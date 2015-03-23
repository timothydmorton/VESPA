.. _api:

API
===

.. module:: vespa

This page details some of the most important methods and classes
provided by the ``vespa`` module.  Please note that this documentation
is a work in progress, and not yet complete.  The simplest entry point
into these calculations is the ``calcfpp`` command-line script, which
creates a :class:`FPPCalculation` object using
:func:`FPPCalculation.from_ini`.  The basic object representing an
eclipse model (either planetary or stellar) is
:class:`EclipsePopulation`, which subclasses the more general
:class:`StarPopulation`, which in turn uses :class:`OrbitPopulation`
or :class:`TripleOrbitPopulation` to simulate randome orbits.

FPPCalculation
-----------------

.. autoclass:: vespa.FPPCalculation
   :members:

Eclipse Populations
-----------------

.. autoclass:: vespa.EclipsePopulation
   :members:

Undiluted Eclipsing Binary
^^^^^^^^^^^^^^^

.. autoclass:: vespa.EBPopulation
   :members:

Hierarchical-triple Eclipsing Binary
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: vespa.HEBPopulation
   :members:

Background Eclipsing Binary
^^^^^^^^^^^^^^^^^

.. autoclass:: vespa.BEBPopulation
   :members:

Transiting Planet
^^^^^^^^^^^^^^^^^^

.. autoclass:: vespa.PlanetPopulation
   :members:

General Star Populations
-----------------

.. autoclass:: vespa.StarPopulation
   :members:
