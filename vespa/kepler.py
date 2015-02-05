from __future__  import print_function, division
import numpy as np
import pandas as pd


def get_rowefit(koi):
    folder = '%s/koi%i.n' % (ROWEFOLDER,ku.koiname(koi,star=True,koinum=True))    
    num = np.round(ku.koiname(koi,koinum=True) % 1 * 100)    
    rowefitfile = '%s/n%i.dat' % (folder,num)
    try:
        return pd.read_table(rowefitfile,index_col=0,usecols=(0,1,3),
                             names=['par','val','a','err','c'],
                             delimiter='\s+')
    except IOError:
        raise MissingKOIError('{} does not exist.'.format(rowefitfile))

class KeplerTransitsignal(fpp.Transitsignal):
    def __init__(self,koi,mcmc=True,maxslope=None,refit_mcmc=False,
                 photfile=None,photcols=(0,1),Tdur=None,ror=None,P=None,**kwargs):
        self.folder = '%s/koi%i.n' % (ROWEFOLDER,ku.koiname(koi,star=True,koinum=True))
        num = np.round(ku.koiname(koi,koinum=True) % 1 * 100)

        if photfile is None:
            self.lcfile = '%s/tremove.%i.dat' % (self.folder,num)
            if not os.path.exists(self.lcfile):
                raise MissingKOIError('{} does not exist.'.format(self.lcfile))
            logging.debug('Reading photometry from {}'.format(self.lcfile))

            #break if photometry file is empty
            if os.stat(self.lcfile)[6]==0:
                raise EmptyPhotometryError('{} photometry file ({}) is empty'.format(ku.koiname(koi),
                                                                                      self.lcfile))

            lc = pd.read_table(self.lcfile,names=['t','f','df'],
                                                      delimiter='\s+')
            self.ttfile = '%s/koi%07.2f.tt' % (self.folder,ku.koiname(koi,koinum=True))
            self.has_ttvs = os.path.exists(self.ttfile)
            if self.has_ttvs:            
                if os.stat(self.ttfile)[6]==0:
                    self.has_ttvs = False
                    logging.warning('TTV file exists for {}, but is empty.  No TTVs applied.'.format(ku.koiname(koi)))
                else:
                    logging.debug('Reading transit times from {}'.format(self.ttfile))
                    tts = pd.read_table(self.ttfile,names=['tc','foo1','foo2'],delimiter='\s+')

            self.rowefitfile = '%s/n%i.dat' % (self.folder,num)

            self.rowefit = pd.read_table(self.rowefitfile,index_col=0,usecols=(0,1,3),
                                        names=['par','val','a','err','c'],
                                        delimiter='\s+')

            logging.debug('JRowe fitfile: {}'.format(self.rowefitfile))

            P = self.rowefit.ix['PE1','val']
            RR = self.rowefit.ix['RD1','val']
            aR = (self.rowefit.ix['RHO','val']*G*(P*DAY)**2/(3*np.pi))**(1./3)
            cosi = self.rowefit.ix['BB1','val']/aR
            Tdur = P*DAY/np.pi*np.arcsin(1/aR * (((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))**(0.5))/DAY

            if 1/aR * (((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))**(0.5) > 1:
                logging.warning('arcsin argument in Tdur calculation > 1; setting to 1 for purposes of rough Tdur calculation...')
                Tdur = P*DAY/np.pi*np.arcsin(1)/DAY

            if (1+RR) < (self.rowefit.ix['BB1','val']):
                #Tdur = P*DAY/np.pi*np.arcsin(1/aR * (((1+RR)**2 - (aR*0)**2)/(1 - 0**2))**(0.5))/DAY/2.
                raise BadRoweFitError('best-fit impact parameter ({:.2f}) inconsistent with best-fit radius ratio ({}).'.format(self.rowefit.ix['BB1','val'],RR))

            if RR < 0:
                raise BadRoweFitError('{0} has negative RoR ({1}) from JRowe MCMC fit'.format(ku.koiname(koi),RR))
            if RR > 1:
                raise BadRoweFitError('{0} has RoR > 1 ({1}) from JRowe MCMC fit'.format(ku.koiname(koi),RR))            
            if aR < 1:
                raise BadRoweFitError('{} has a/Rstar < 1 ({}) from JRowe MCMC fit'.format(ku.koiname(koi),aR))


            self.P = P
            self.aR = aR
            self.Tdur = Tdur
            self.epoch = self.rowefit.ix['EP1','val'] + 2504900

            logging.debug('Tdur = {:.2f}'.format(self.Tdur))
            logging.debug('aR={0}, cosi={1}, RR={2}'.format(aR,cosi,RR))
            logging.debug('arcsin arg={}'.format(1/aR * (((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))**(0.5)))
            logging.debug('inside sqrt in arcsin arg={}'.format((((1+RR)**2 - (aR*cosi)**2)/(1 - cosi**2))))
            logging.debug('best-fit impact parameter={:.2f}'.format(self.rowefit.ix['BB1','val']))

            lc['t'] += (2450000+0.5)
            lc['f'] += 1 - self.rowefit.ix['ZPT','val']

            if self.has_ttvs:
                tts['tc'] += 2504900

            ts = pd.Series()
            fs = pd.Series()
            dfs = pd.Series()

            if self.has_ttvs:
                for t0 in tts['tc']:
                    t = lc['t'] - t0
                    ok = np.absolute(t) < 2*self.Tdur
                    ts = ts.append(t[ok])
                    fs = fs.append(lc['f'][ok])
                    dfs = dfs.append(lc['df'][ok])
            else:
                center = self.epoch % self.P
                t = np.mod(lc['t'] - center + self.P/2,self.P) - self.P/2
                ok = np.absolute(t) < 2*self.Tdur
                ts = t[ok]
                fs = lc['f'][ok]
                dfs = lc['df'][ok]

            logging.debug('{0}: has_ttvs is {1}'.format(koi,self.has_ttvs))
            logging.debug('{} light curve points used'.format(ok.sum()))


            if maxslope is None:
                #set maxslope using duration
                maxslope = max(Tdur*24/0.5 * 2, 30) #hardcoded in transitFPP as default=30

            p0 = [Tdur,RR**2,3,0]
            self.p0 = p0
            logging.debug('initial trapezoid parameters guess: {}'.format(p0))
            fpp.Transitsignal.__init__(self,np.array(ts),np.array(fs),np.array(dfs),p0=p0,
                                       name=ku.koiname(koi),
                                       P=P,maxslope=maxslope)
        else:
            if P is None:
                P = ku.DATA[koi]['koi_period']
            ts,fs = np.loadtxt(photfile,usecols=photcols,unpack=True)
            if Tdur is not None and ror is not None:
                p0 = [Tdur,ror**2,3,0]
            else:
                p0 = None
            fpp.Transitsignal.__init__(self,ts,fs,name=ku.koiname(koi),
                                       P=P,
                                       maxslope=maxslope,p0=p0)
        
        if mcmc:
            self.MCMC(refit=refit_mcmc)

        if self.hasMCMC and not self.fit_converged:
            logging.warning('Trapezoidal MCMC fit did not converge for {}.'.format(self.name))


    def MCMC(self,**kwargs):
        folder = '%s/%s' % (CHAINSDIR,self.name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        fpp.Transitsignal.MCMC(self,savedir=folder,**kwargs)

