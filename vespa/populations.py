from __future__ import print_function, division

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .transit_basic import occultquad, ldcoeffs, minimum_inclination
from .transit_basic import MAInterpolationFunction

from starutils.populations import StarPopulation, MultipleStarPopulation
from starutils.populations import ColormatchMultipleStarPopulation
from starutils.utils import draw_eccs, semimajor, withinroche
from starutils.utils import mult_masses
from starutils.utils import RAGHAVAN_LOGPERKDE

SHORT_MODELNAMES = {'Planets':'pl',
                    'EBs':'eb',
                    'HEBs':'heb',
                    'BEBs':'beb',
                    'Blended Planets':'bpl',
                    'Specific BEB':'sbeb',
                    'Specific HEB':'sheb'}
                        
INV_SHORT_MODELNAMES = {v:k for k,v in SHORT_MODELNAMES.iteritems()}

import astropy.constants as const
AU = const.au.cgs.value
RSUN = const.R_sun.cgs.value
G = const.G.cgs.value

class EclipsePopulation(StarPopulation):
    def __init__(self, stars=None, P=None, model='',
                 priorfactors=None, lhoodcachefile=None,
                 orbpop=None,
                 **kwargs):
        """Base class for populations of eclipsing things.

        stars DataFrame must have the following columns:
        'M_1', 'M_2', 'R_1', 'R_2', 'u1_1', 'u1_2', 'u2_1', 'u2_2', and 
        either the P keyword argument provided or a `period column, as 
        well as the eclipse parameters: 'inc', 'ecc', 'w', 'dpri', 
        'dsec', 'b_sec', 'b_pri', 'fluxfrac_1', 'fluxfrac_2'

        For some functionality, also needs to have trapezoid fit 
        parameters in DataFrame
        
        """
        
        self.P = P
        self.model = model
        if priorfactors is None:
            priorfactors = {}
        self.priorfactors = priorfactors
        self.lhoodcachefile = lhoodcachefile
        
        self.is_specific = False
        self.is_ruled_out = False

        try:
            self.modelshort = SHORT_MODELNAMES[model]
            
            #add index if specific model is indexed
            if hasattr(self,'index'):
                self.modelshort += '-{}'.format(self.index)

        except KeyError:
            raise KeyError('No short name for model: %s' % model)

        StarPopulation.__init__(self, stars=stars, orbpop=orbpop)
        
        if stars is not None:
            if len(self.stars)==0:
                raise EmptyPopulationError('Zero elements in {} population'.format(model))

        #This will throw error if trapezoid fits not done
        self.make_kdes()

    def fit_trapezoids(self):
        pass

    def save_hdf(self, filename, path='', properties=None, **kwargs):
        if properties is None:
            properties = {}
        
        for prop in []:
            properties[prop] = getattr(self,prop)

        StarPopulation.save_hdf(self, filename, path=path,
                                properties=properties, **kwargs)

    def load_hdf(self): #perhaps this doesn't need to be written?
        pass

class HEBPopulation(EclipsePopulation, ColormatchMultipleStarPopulation):
    def __init__(self, filename=None, period=None, mags=None, colors=['JK'], 
                 mass=None, age=None, feh=None, starfield=None,
                 band='Kepler', modelname='HEBs', f_triple=0.12, n=2e4,
                 **kwargs):
        """Population of HEBs

        If file is passed, population is loaded from .h5 file.

        If file not passed, then a population will be generated.
        If mdist, agedist, and fehdist are passed, then the primary of
        the population will be generated according to those distributions.
        If distributions are not passed, then populations should be generated
        in order to match colors.

        mass is primary mass.  mass, age, and feh can be distributions
        (or tuples)

        kwargs passed to ``ColormatchMultipleStarPopulation`` 
        """

        self.period = period
        self.band = band
        self.f_triple = f_triple

        if filename is not None:
            self.load_hdf(filename)

        else:
            #if provided, period_short is the observed period of the eclipse
            pop_kwargs = {'mags':mags, 'colors':colors,
                          'starfield':starfield, 'f_triple':1,
                          'period_short':self.period}

            pop = ColormatchMultipleStarPopulation(mags=mags,
                                                   colors=colors, mA=mass,
                                                   age=age, feh=feh,
                                                   starfield=starfield,
                                                   f_triple=1, period_short=self.period,
                                                   n=n)
            
            s = pop.stars

            #calculate limb-darkening coefficients
            u1A, u2A = ldcoeffs(s['Teff_A'], s['logg_A'])
            u1B, u2B = ldcoeffs(s['Teff_B'], s['logg_B'])
            u1C, u2C = ldcoeffs(s['Teff_C'], s['logg_C'])


            #calculate eclipses.  In the MultipleStarPopulation, stars '_B' and '_C'
            # are always the ones eclipsing each other.

            #Need to change this to return the right number/stars....

            inds, df, (prob,dprob) = calculate_eclipses(s['mass_B'], s['mass_C'],
                                                        s['radius_B'], s['radius_C'],
                                                        s['{}_mag_B'.format(band)], 
                                                        s['{}_mag_C'.format(band)],
                                                        u11s=u1B, u21s=u2B,
                                                        u12s=u1C, u22s=u2C, band=band,
                                                        period=self.period, calc_mininc=True,
                                                        return_indices=True)

            logging.debug('{} nans in dpri'.format(np.isnan(df['dpri']).sum()))
            logging.debug('{} nans in dsec'.format(np.isnan(df['dsec']).sum()))

            self.eclipse_pars = df
            self.eclipse_prob = prob
            self.eclipse_dprob = dprob
            for col in df.columns:
                s[col] = df[col]
            
            
                                                  


    def save_hdf(self, filename, path='', properties=None, **kwargs):
        if properties is None:
            properties = {}
        
        for prop in []:
            properties[prop] = getattr(self,prop)

        EclipsePopulation.save_hdf(self, filename, path=path,
                                   properties=properties, **kwargs)


def calculate_eclipses(M1s, M2s, R1s, R2s, mag1s, mag2s,
                       u11s=None, u21s=None, u12s=None, u22s=None,
                       Ps=None, period=None, logperkde=RAGHAVAN_LOGPERKDE,
                       incs=None, eccs=None, band='i',
                       mininc=None, maxecc=0.97, verbose=False,
                       return_probability_only=False, return_indices=False,
                       calc_mininc=True, MAfn=None):
    """Returns random eclipse parameters for provided inputs

    If single period desired, pass 'period' keyword.
    """
    if MAfn is None:
        MAfn = MAInterpolationFunction(nzs=200,nps=400,pmin=0.007,pmax=1/0.007)

    M1s = np.atleast_1d(M1s)
    M2s = np.atleast_1d(M2s)
    R1s = np.atleast_1d(R1s)
    R2s = np.atleast_1d(R2s)
    mag1s = np.atleast_1d(mag1s)
    mag2s = np.atleast_1d(mag2s)
    u11s = np.atleast_1d(u11s)
    u12s = np.atleast_1d(u12s)
    u21s = np.atleast_1d(u21s)
    u22s = np.atleast_1d(u22s)

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
            Ps[tooclose] = 10**(logperkde.rvs(ntooclose))
            if simeccs:
                eccs[tooclose] = draw_eccs(ntooclose,Ps[tooclose])
            semimajors[tooclose] = semimajor(Ps[tooclose],M1s[tooclose]+M2s[tooclose])*AU
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
    if mininc is None and calc_mininc:
        mininc = minimum_inclination(Ps, M1s, M2s, R1s, R2s)

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


    b_tras = semimajors*np.cos(incs)/(R_large*RSUN) * (1-eccs**2)/(1 + eccs*np.sin(ws))
    b_occs = semimajors*np.cos(incs)/(R_large*RSUN) * (1-eccs**2)/(1 - eccs*np.sin(ws))

    b_tras[tooclose] = np.inf
    b_occs[tooclose] = np.inf

    ks = R_small/R_large
    Rtots = (R_small + R_large)/R_large
    tra = (b_tras < Rtots)
    occ = (b_occs < Rtots)
    nany = (tra | occ).sum()
    peb = nany/float(n)
    if return_probability_only:
        return prob,prob*np.sqrt(nany)/n


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
        logging.error('b_tra: {}'.format(b_tra[wbad]))
        logging.error('b_occ: {}'.format(b_occ[wbad]))
        logging.error('T14_tra: {}'.format(T14_tra[wbad]))
        logging.error('T14_occ: {}'.format(T14_occ[wbad]))
        logging.error('under sqrt (tra): {}'.format((1+k[wbad])**2 - b_tra[wbad]**2))
        logging.error('under sqrt (occ): {}'.format((1+k[wbad])**2 - b_occ[wbad]**2))
        logging.error('eccsq: {}'.format(ecc[wbad]**2))
        logging.error('a in Rsun: {}'.format(a[wbad]/RSUN))
        logging.error('R_large: {}'.format(R_large[wbad]))
        logging.error('R_small: {}'.format(R_small[wbad]))
        logging.error('P: {}'.format(P[wbad]))
        logging.error('total M: {}'.format(M1[w]+M2[wbad]))

    T14_tra[(np.isnan(T14_tra))] = 0
    T23_tra[(np.isnan(T23_tra))] = 0
    T14_occ[(np.isnan(T14_occ))] = 0
    T23_occ[(np.isnan(T23_occ))] = 0

    #calling mandel-agol
    ftra = MAfn(k,b_tra,u11,u21)
    focc = MAfn(1/k,b_occ/k,u12,u22)

    #fix those with k or 1/k out of range of MAFN....or do it in MAfn eventually?
    wtrabad = np.where((k < MAfn.pmin) | (k > MAfn.pmax))
    woccbad = np.where((1/k < MAfn.pmin) | (1/k > MAfn.pmax))
    for ind in wtrabad[0]:
        ftra[ind] = occultquad(b_tra[ind],u11[ind],u21[ind],k[ind])
    for ind in woccbad[0]:
        focc[ind] = occultquad(b_occ[ind]/k[ind],u12[ind],u22[ind],1/k[ind])

    F1 = 10**(-0.4*mag1) + switched*10**(-0.4*mag2)
    F2 = 10**(-0.4*mag2) + switched*10**(-0.4*mag1)

    dtra = 1-(F2 + F1*ftra)/(F1+F2)
    docc = 1-(F1 + F2*focc)/(F1+F2)

    totmag = -2.5*np.log10(F1+F2)

    #wswitched = where(switched)
    dtra[switched],docc[switched] = (docc[switched],dtra[switched])
    T14_tra[switched],T14_occ[switched] = (T14_occ[switched],T14_tra[switched])
    T23_tra[switched],T23_occ[switched] = (T23_occ[switched],T23_tra[switched])
    b_tra[switched],b_occ[switched] = (b_occ[switched],b_tra[switched])
    #mag1[wswitched],mag2[wswitched] = (mag2[wswitched],mag1[wswitched])
    F1[switched],F2[switched] = (F2[switched],F1[switched])
    u11[switched],u12[switched] = (u12[switched],u11[switched])
    u21[switched],u22[switched] = (u22[switched],u21[switched])

    dtra[(np.isnan(dtra))] = 0
    docc[(np.isnan(docc))] = 0

    df =  pd.DataFrame({'{}_mag_tot'.format(band) : totmag,
                        'P':P, 'ecc':ecc, 'inc':inc, 'w':w,
                        'dpri':dtra, 'dsec':docc,
                        'T14_pri':T14_tra, 'T23_pri':T23_tra,
                        'T14_sec':T14_occ, 'T23_sec':T23_occ,
                        'b_pri':b_tra, 'b_sec':b_occ,
                        '{}_mag_1'.format(band) : mag1,
                        '{}_mag_2'.format(band) : mag2,
                        'fluxfrac_1':F1/(F1+F2),
                        'fluxfrac_2':F2/(F1+F2),
                        'switched':switched,
                        'u1_1':u11, 'u2_1':u21, 'u1_2':u12, 'u2_2':u22})

    if return_indices:
        return wany, df, (prob, prob*np.sqrt(nany)/n)
    else:
        return df, (prob, prob*np.sqrt(nany)/n)
