from __future__ import print_function, division

import os
on_rtd = os.environ.get('READTHEDOCS') == 'True'

if not on_rtd:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from scipy.interpolate import UnivariateSpline as interpolate
    from scipy.integrate import quad
else:
    np, pd, plt = (None, None, None)
    interpolate, quad = (None, None)

from ..plotutils import setfig
from ..hashutils import hashcombine, hasharray

from .constraints import FunctionLowerLimit

class ContrastCurve(object):
    """Object representing an imaging contrast curve

    Usually accessed via :class:`ContrastCurveFromFile`
    and then applied using :class:`ContrastCurveConstraint`,
    e.g., through :func:`StarPopulation.apply_cc`.

    :param rs:
        Angular separation from target star, in arcsec.

    :param dmags:
        Magnitude contrast.

    :param band:
        Photometric bandpass in which observation is taken.

    :param mag:
        Magnitude of central star (rarely used?)

    :param name:
        Name; e.g., "PHARO J-band", "Keck AO", etc.
        Should be a decent label.


    """
    def __init__(self,rs,dmags,band,mag=None,name=None):

        #if band=='K' or band=="K'":
        #    band = 'Ks'

        rs = np.atleast_1d(rs)
        dmags = np.atleast_1d(dmags)
        self.rs = rs
        self.dmags = dmags
        self.band = band
        self.mag = mag
        self.contrastfn = interpolate(rs,dmags,s=0)
        self.rmax = rs.max()
        self.rmin = rs.min()
        if name is None:
            self.name = '%s band' % self.band
        else:
            self.name = name

    def plot(self,fig=None,**kwargs):
        setfig(fig)
        plt.plot(self.rs,self.dmags,**kwargs)
        plt.title('%s band contrast curve' % self.band)
        plt.gca().invert_yaxis()
        plt.xlabel('Separation [arcsec]')
        plt.ylabel('$\Delta %s$' % self.band)

    def __eq__(self,other):
        return hash(self)==hash(other)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __hash__(self):
        return hashcombine(hasharray(self.rs),
                           hasharray(self.dmags),
                           self.band,
                           self.mag)
    def __call__(self,r):
        r = np.atleast_1d(r)
        dmags = np.atleast_1d(self.contrastfn(r))
        dmags[r >= self.rmax] = self.contrastfn(self.rmax)
        dmags[r < self.rmin] = 0
        #put something in here to "extend" beyond rmax?
        return dmags

    def __add__(self,other):
        if type(other) not in [type(1),type(1.),type(self)]:
            raise ValueError('Can only add a number or another ContrastCurve.')
        if type(other) in [type(1),type(1.)]:
            dmags = self.dmags + other
            return ContrastCurve(self.rs,dmags,self.band,self.mag)

    def __repr__(self):
        return '<%s: %s>' % (type(self),self.name)

    def power(self,floor=10,rmin=0.1,use_quad=False):
        if use_quad:
            return quad(self,rmin,self.rmax)[0]/((self.rmax-rmin)*floor)
        else:
            rs = np.linspace(rmin,self.rmax,100)
            return np.trapz(self(rs),rs)

class ContrastCurveConstraint(FunctionLowerLimit):
    def __init__(self,rs,dmags,cc,name='CC',**kwargs):
        self.rs = rs
        self.dmags = dmags
        self.cc = cc
        FunctionLowerLimit.__init__(self,rs,dmags,cc,name=name,**kwargs)

    def __str__(self):
        return '%s contrast curve' % self.name

    def update_rs(self,rs):
        self.rs = rs
        FunctionLowerLimit.__init__(self,rs,self.dmags,self.cc,name=self.name)
        logging.info('%s updated with new rsky values.' % self.name)

class ContrastCurveFromFile(ContrastCurve):
    """A contrast curve derived from a two-column file

    :param filename:
        Filename of contrast curve; first column separation in arcsec,
        second column delta-mag.

    :param band:
        Bandpass of imaging observation.

    :param mas:
        Set to ``True`` if separation is in milliarcsec rather than
        arcsec.

    """
    def __init__(self,filename,band,mag=None, mas=False, **kwargs):
        rs,dmags = np.loadtxt(filename,unpack=True)
        if mas: #convert from milliarcsec
            rs /= 1000.
        ContrastCurve.__init__(self,rs,dmags,band,mag, **kwargs)
        self.filename = filename


class VelocityContrastCurve(object):
    def __init__(self,vs,dmags,band='g'):
        self.vs = vs
        self.dmags = dmags
        self.band = band
        if np.size(vs) > 1:
            self.contrastfn = interpolate(vs,dmags,s=0)
            self.vmax = vs.max()
            self.vmin = vs.min()
        else: #simple case; one v, one dmag
            def cfn(v):
                v = np.atleast_1d(abs(v))
                dmags = np.zeros(v.shape)
                dmags[v>=self.vs] = self.dmags
                dmags[v<self.vs] = 0
                return dmags
            self.contrastfn = cfn
            self.vmax = self.vs
            self.vmin = self.vs

    def __call__(self,v):
        v = np.atleast_1d(np.absolute(v))
        dmags = np.atleast_1d(self.contrastfn(v))
        dmags[v >= self.vmax] = self.contrastfn(self.vmax)
        dmags[v < self.vmin] = 0
        #put something in here to "extend" beyond vmax?
        return dmags

class VelocityContrastCurveConstraint(FunctionLowerLimit):
    def __init__(self,vels,dmags,vcc,name='VCC',**kwargs):
        self.vels = vels
        self.dmags = dmags
        self.vcc = vcc
        FunctionLowerLimit.__init__(self,vels,dmags,vcc,name=name,**kwargs)
