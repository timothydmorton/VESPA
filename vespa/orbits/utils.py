import logging

try:
    import numpy as np
    import numpy.random as rand

    from astropy import units as u
    from astropy.coordinates import SkyCoord,Angle
    from astropy.units.quantity import Quantity
    from astropy import constants as const
    MSUN = const.M_sun.cgs.value
    AU = const.au.cgs.value
    DAY = 86400
    G = const.G.cgs.value
except ImportError:
    np,rand = (None, None)
    u, SkyCoord, Angle, Quantity, Const = (None, None, None, None, None)
    MSUN, AU, DAY, G = (None, None, None, None)

import os
on_rtd = os.environ.get('READTHEDOCS') == 'True'

if not on_rtd:
    from .kepler import Efn #

def semimajor(P,M):
    """
    P, M can be ``Quantity`` objects; otherwise default to day, M_sun
    """
    if type(P) != Quantity:
        P = P*u.day
    if type(M) != Quantity:
        M = M*u.M_sun
    a = ((P/2/np.pi)**2*const.G*M)**(1./3)
    return a.to(u.AU)

def random_spherepos(n):
    """
    Returns SkyCoord object with n positions randomly oriented on the unit sphere.

    :param n: (``int``)
        Number of positions desired.

    :return c:
        ``astropy.coordinates.SkyCoord`` object with random positions 
    """
    signs = np.sign(rand.uniform(-1,1,size=n))
    thetas = Angle(np.arccos(rand.uniform(size=n)*signs),unit=u.rad) #random b/w 0 and 180
    phis = Angle(rand.uniform(0,2*np.pi,size=n),unit=u.rad)
    c = SkyCoord(phis,thetas,1,representation='physicsspherical')
    return c

def orbitproject(x,y,inc,phi=0,psi=0):
    """
    Transform x,y planar coordinates into observer's coordinate frame.

    ``x,y`` are coordinates in ``z=0`` plane (plane of the orbit)

    observer is at ``(inc, phi)`` on celestial sphere (angles in radians);
    ``psi`` is orientation of final ``x-y`` axes about the ``(inc,phi)`` vector.

    Returns ``x,y,z`` values in observer's coordinate frame, where
    ``x,y`` are now plane-of-sky coordinates and ``z`` is along the line of sight.

    :param x,y: (``float`` or array-like)
        Coordinates to transform.

    :param inc: (``float`` or array-like)
        Polar angle(s) of observer (where ``inc=0`` corresponds to north pole
        of original ``x-y`` plane).  This angle is the same as standard "inclination."

    :param phi: (``float`` or array-like, optional)
        Azimuthal angle of observer around ``z`` -axis

    :param psi: (``float`` or array-like, optional)
        Orientation of final observer coordinate frame (azimuthal around
        ``(inc,phi)`` vector.

    :return x,y,z: (``ndarray``)
        Coordinates in observers' frames.  ``x,y`` in "plane of sky" and ``z``
        along line of sight.
    """

    x2 = x*np.cos(phi) + y*np.sin(phi)
    y2 = -x*np.sin(phi) + y*np.cos(phi)
    z2 = y2*np.sin(inc)
    y2 = y2*np.cos(inc)

    xf = x2*np.cos(psi) - y2*np.sin(psi)
    yf = x2*np.sin(psi) + y2*np.cos(psi)

    return (xf,yf,z2)

def orbit_posvel(Ms,eccs,semimajors,mreds,obspos=None):
    """
    Returns positions in projected AU and velocities in km/s for given mean anomalies.

    Returns 3-D positions and velocities as SkyCoord objects, in
    "observer" reference frame.  Uses
    :func:`kepler.Efn` to calculate eccentric anomalies using interpolation.    

    :param Ms,eccs,semimajors,mreds: (``float`` or array-like)
        Mean anomalies, eccentricities, semimajor axes [AU], reduced masses [Msun].

    :param obspos: (``None``, ``(x,y,z)`` ``tuple`` or ``SkyCoord`` object)
        Locations of observers for which to return coordinates.
        If ``None`` then populate randomly on sphere.  If ``(x,y,z)`` or
        ``SkyCoord`` object provided, then use those.

    :returns pos,vel:
        ``SkyCoord`` Objects representing the positions and velocities,
        the coordinates
        of which are ``Quantity`` objects that have units.  Positions are in
        projected AU and velocities in km/s.
        
    """

    Es = Efn(Ms,eccs) #eccentric anomalies by interpolation

    rs = semimajors*(1-eccs*np.cos(Es))
    nus = 2 * np.arctan2(np.sqrt(1+eccs)*np.sin(Es/2),np.sqrt(1-eccs)*np.cos(Es/2))

    xs = semimajors*(np.cos(Es) - eccs)         #AU
    ys = semimajors*np.sqrt(1-eccs**2)*np.sin(Es)  #AU
    
    Edots = np.sqrt(G*mreds*MSUN/(semimajors*AU)**3)/(1-eccs*np.cos(Es))
        
    xdots = -semimajors*AU*np.sin(Es)*Edots/1e5  #km/s
    ydots = semimajors*AU*np.sqrt(1-eccs**2)*np.cos(Es)*Edots/1e5 # km/s
        
    n = np.size(xs)

    orbpos = SkyCoord(xs,ys,0*u.AU,representation='cartesian',unit='AU')
    orbvel = SkyCoord(xdots,ydots,0*u.km/u.s,representation='cartesian',unit='km/s')
    if obspos is None:
        obspos = random_spherepos(n) #observer position
    if type(obspos) == type((1,2,3)):
        obspos = SkyCoord(obspos[0],obspos[1],obspos[2],
                          representation='cartesian').represent_as('physicsspherical')

    if not hasattr(obspos,'theta'): #if obspos not physics spherical, make it 
        obspos = obspos.represent_as('physicsspherical')
        
    #random orientation of the sky 'x-y' coordinates
    psi = rand.random(n)*2*np.pi  

    #transform positions and velocities into observer coordinates
    x,y,z = orbitproject(orbpos.x,orbpos.y,obspos.theta,obspos.phi,psi)
    vx,vy,vz = orbitproject(orbvel.x,orbvel.y,obspos.theta,obspos.phi,psi)

    return (SkyCoord(x,y,z,representation='cartesian'),
            SkyCoord(vx,vy,vz,representation='cartesian')) 
