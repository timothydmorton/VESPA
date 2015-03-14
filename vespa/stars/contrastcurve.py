from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline as interpolate
from scipy.integrate import quad

from plotutils import setfig
from hashutils import hashcombine, hasharray

from .constraints import FunctionLowerLimit

class ContrastCurve(object):
    def __init__(self,rs,dmags,band,mag=None,name=None):
        """band is self-explanatory; 'mag' is mag of the primary in 'band' """
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
    def __init__(self,filename,band,mag=None):
        rs,dmags = np.loadtxt(filename,unpack=True)
        if rs[0] > 2:
            rs /= 1000.
        ContrastCurve.__init__(self,rs,dmags,band,mag)
        self.filename = filename


class VelocityContrastCurveConstraint(FunctionLowerLimit):
    def __init__(self,vels,dmags,vcc,name='VCC',**kwargs):
        self.vels = vels
        self.dmags = dmags
        self.vcc = vcc
        FunctionLowerLimit.__init__(self,vels,dmags,vcc,name=name,**kwargs)
