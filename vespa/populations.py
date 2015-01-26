from __future__ import print_function, division

import logging

import numpy as np
import matplotlib.pyplot as plt

from .transit_basic import impact_parameter

from starutils.populations import StarPopulation
from starutils.utils import draw_eccs, semimajor, withinroche
from starutils.utils import RAGHAVAN_LOGPERKDE

SHORT_MODELNAMES = {'Planets':'pl',
                    'EBs':'eb',
                    'HEBs':'heb',
                    'BEBs':'beb',
                    'Blended Planets':'bpl',
                    'Specific BEB':'sbeb',
                    'Specific HEB':'sheb'}
                        
INV_SHORT_MODELNAMES = {v:k for k,v in SHORT_MODELNAMES.iteritems()}

from astropy.constants as const
AU = const.au.cgs.value

class EclipsePopulation(StarPopulation):
    def __init__(self, stars, P=None, model='',
                 priorfactors=None, lhoodcachefile=None,
                 **kwargs):
        """Base class for populations of eclipsing things.


        stars DataFrame must have parameters describing orbit/eclipse:
        'P', 'M1', 'M2', 'R1', 'R2', 'inc', 'ecc', 'w', 'dpri', 
        'dsec', 'b_sec', 'b_pri', 'fluxfrac1', 'fluxfrac2', 
        'u11', 'u12', 'u21', 'u22'
        
        """
        
        self.P = P
        self.model = model

        try:
            self.modelshort = SHORT_MODELNAMES[model]
            
            #add index if specific model is indexed
            if hasattr(self,'index'):
                self.modelshort += '-{}'.format(self.index)

        except KeyError:
            raise KeyError('No short name for model: %s' % model)

        if priorfactors is None:
            priorfactors = {}
        self.priorfactors = priorfactors

        self.lhoodcacefile = lhoodcachefile

        self.is_specific = False

        StarPopulation.__init__(self, stars)
        
        self.is_ruled_out = False

        if len(self.stars)==0:
            raise EmptyPopulationError('Zero elements in {} population'.format(model))

        self.make_kdes()


def calculate_eclipses(M1s, M2s, R1s, R2s, mag1s, mag2s,
                       u11s=None, u21s=None, u12s=None, u22s=None,
                       Ps=None, period=None, logperkde=RAGHAVAN_LOGPERKDE,
                       incs=None, eccs=None, band='i',
                       mininc=None, maxecc=0.97, verbose=False,
                       return_probability=False):
    """Returns random eclipse parameters for provided inputs

    """


    n = np.size(M1s)
    
    #a bit clunky here, but works.
    simPs = False
    if period:
        Ps = np.ones(n)*period
    else:
        if Ps is None:
            Ps = 10**(logperkde.rvs(n))
            simPs = True
    simeccs = False
    if eccs is None:
        if not simPs and period is not None:
            eccs = draw_eccs(n,period,maxecc=maxecc)
        else:
            eccs = draw_eccs(n,Ps,maxecc=maxecc)
        simeccs = True

    semimajors = semimajor(Ps, M1s+M2s)*AU #in AU

    #check to see if there are simulated instances that are
    # too close; i.e. periastron sends secondary within roche 
    # lobe of primary
    tooclose = withinroche(semimajors*(1-eccs)/AU,M1s,R1s,M2s,R2s)
    ntooclose = tooclose.sum()
    tries = 0
    maxtries=5
    if simPs:
        while ntooclose > 0:
            lastntooclose=ntooclose
            Ps[tooclose] = 10**(logperkde.draw(ntooclose))
            if simeccs:
                eccs[wtooclose] = draw_eccs(ntooclose,Ps[tooclose])
            semimajors[wtooclose] = semimajor(Ps[tooclose],M1s[tooclose]+M2s[tooclose])*AU
            tooclose = withinroche(semimajors*(1-eccs)/AU,M1s,R1s,M2s,R2s)
            ntooclose = tooclose.sum()
            if ntooclose==lastntooclose:   #prevent infinite loop
                tries += 1
                if tries > maxtries:
                    if verbose:
                        logging.info('{} binaries are "too close"; gave up trying to fix.'.format(ntooclose))
                    break                       
    else:
        while ntooclose > 0:
            lastntooclose=ntooclose
            if simeccs:
                eccs[tooclose] = draw_eccs(ntooclose,Ps[tooclose])
            semimajors[tooclose] = semimajor(Ps[tooclose],M1s[tooclose]+M2s[tooclose])*AU
            #wtooclose = where(semimajors*(1-eccs) < 2*(R1s+R2s)*RSUN)
            tooclose = withinroche(semimajors*(1-eccs)/AU,M1s,R1s,M2s,R2s)
            ntooclose = tooclose.sum()
            if ntooclose==lastntooclose:   #prevent infinite loop
                tries += 1
                if tries > maxtries:
                    if verbose:
                        logging.info('{} binaries are "too close"; gave up trying to fix.'.format(ntooclose))
                    break                       

    #randomize inclinations, either full range, or within restricted range
    if incs is None:
        if mininc is None:
            incs = np.arccos(np.random.random(n)) #random inclinations in radians
        else:
            incs = np.arccos(np.random.random(n)*np.cos(mininc*np.pi/180))
    if mininc:
        prob = np.cos(mininc*np.pi/180)
    else:
        prob = 1

    ws = np.random.random(n)*2*np.pi

    switched = (R2s > R1s)
    R_large = switched*R2s + ~switched*R1s
    R_small = switched*R1s + ~switched*R2s


    #b_tras, b_occs = impact_parameter(semimajors/AU, R_large, incs, ecc=eccs, w=ws,
    #                                  return_occ=True)
    b_tras = semimajors*np.cos(incs)/(R_large*RSUN) * (1-eccs**2)/(1 + eccs*np.sin(ws))
    b_occs = semimajors*np.cos(incs)/(R_large*RSUN) * (1-eccs**2)/(1 - eccs*np.sin(ws))

    b_tras[tooclose] = inf
    b_occs[tooclose] = inf

    ks = R_small/R_large
    Rtots = (R_small + R_large)/R_large
    tra = (b_tras < Rtots)
    occ = (b_occs < Rtots)
    nany = (tra | occ).sum()
    peb = nany/float(n)
    if return_probability:
        return prob*peb,prob*np.sqrt(nany)/n


    i = (tra | occ)
    wany = np.where(i)
    P,M1,M2,R1,R2,mag1,mag2,inc,ecc,w = Ps[i],M1s[i],M2s[i],R1s[i],R2s[i],\
        mag1s[i],mag2s[i],incs[i]*180/np.pi,eccs[i],ws[i]*180/np.pi
    a = semimajors[i]  #in cm already
    b_tra = b_tras[i]
    b_occ = b_occs[i]
    u11 = u11s[i]
    u21 = u21s[i]
    u12 = u12s[i]
    u22 = u22s[i]
   
    
    switched = (R2 > R1)
    R_large = switched*R2 + ~switched*R1
    R_small = switched*R1 + ~switched*R2
    k = R_small/R_large
    
    #calculate durations
    T14_tra = P/np.pi*np.arcsin(R_large*RSUN/a * np.sqrt((1+k)**2 - b_tra**2)/np.sin(inc*np.pi/180)) *\
        np.sqrt(1-ecc**2)/(1+ecc*np.sin(w*np.pi/180)) #*24*60
    T23_tra = P/np.pi*np.arcsin(R_large*RSUN/a * np.sqrt((1-k)**2 - b_tra**2)/np.sin(inc*np.pi/180)) *\
        np.sqrt(1-ecc**2)/(1+ecc*np.sin(w*np.pi/180)) #*24*60
    T14_occ = P/np.pi*np.arcsin(R_large*RSUN/a * np.sqrt((1+k)**2 - b_occ**2)/np.sin(inc*np.pi/180)) *\
        np.sqrt(1-ecc**2)/(1-ecc*np.sin(w*np.pi/180)) #*24*60
    T23_occ = P/np.pi*np.arcsin(R_large*RSUN/a * np.sqrt((1-k)**2 - b_occ**2)/np.sin(inc*np.pi/180)) *\
        np.sqrt(1-ecc**2)/(1-ecc*np.sin(w*np.pi/180)) #*24*60
    
    bad = (np.isnan(T14_tra) & np.isnan(T14_occ))
    if bad.sum() > 0:
        logging.error('Something snuck through with no eclipses!')
        logging.error('k: {}'.format(k[wbad]))
        logging.error('b_tra:',b_tra[wbad]
        logging.error('b_occ:',b_occ[wbad]
        logging.error('T14_tra:',T14_tra[wbad]
        logging.error('T14_occ:',T14_occ[wbad]
        logging.error('under sqrt (tra):',(1+k[wbad])**2 - b_tra[wbad]**2
        logging.error('under sqrt (occ):',(1+k[wbad])**2 - b_occ[wbad]**2
        logging.error('ecc:',ecc[wbad]**2
        logging.error('a in Rsun:',a[wbad]/RSUN
        logging.error('R_large:',R_large[wbad]
        logging.error('R_small:',R_small[wbad]
        logging.error('P:',P[wbad]
        logging.error('total M:',M1[w]+M2[wbad]
        
    
    
