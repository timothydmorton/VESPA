.. _orbits:

Orbits
=============

.. module:: vespa.orbits

If they represent binary or triple star systems,
:class:`vespa.stars.StarPopulation` objects are created with a large
population of randomized orbits. This is done using
the :class:`OrbitPopulation` and :class:`TripleOrbitPopulation`
objects.

The engine that makes it possible to initialize large numbers of
random orbital positions nearly instantaneously is
the :func:`kepler.Efn` function (as used
by :func:`utils.orbit_posvel`), which uses a precomputed grid to
interpolate the solutions to Kepler's equation for a given mean
anomaly and eccentricity (or arrays thereof).

The final coordinate system of these populations is
"observer-oriented," with the ``z`` axis along the line of sight, and
the ``x-y`` plane being the plane of the sky. Practically, this is
accomplished by first simulating all the random orbits in the ``x-y``
plane, and then "observing" them from lines of sight randomly oriented
on the unit sphere, and projecting appropriately.

Coordinates are handled using :class:`astropy.coordinates.SkyCoord`
objects.

Orbit Populations
---------------

.. autoclass:: vespa.orbits.populations.OrbitPopulation
  :members:
	       
.. autoclass:: vespa.orbits.populations.TripleOrbitPopulation
  :members:
	       
Utility Functions
---------------

The following functions are used in the creation
of :class:`OrbitPopulation` objects. :func:`kepler.Efn` is used for
instanteous solution of Kepler's equation (via interpolation),
and :func:`utils.orbit_posvel` does the projecting of random orbits
into 3-d Cartesian coordinates, assisted by
:func:`utils.orbitproject` and :func:`utils.random_spherepos`.

.. autofunction:: vespa.orbits.kepler.Efn
      
.. autofunction:: vespa.orbits.utils.orbit_posvel

.. autofunction:: vespa.orbits.utils.orbitproject

.. autofunction:: vespa.orbits.utils.random_spherepos	      
