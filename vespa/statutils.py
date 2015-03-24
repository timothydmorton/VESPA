from __future__ import division, print_function

try:
    import numpy as np
except ImportError:
    np = None
    
def kdeconf(kde,conf=0.683,xmin=None,xmax=None,npts=500,
            shortest=True,conftol=0.001,return_max=False):
    """
    Returns desired confidence interval for provided KDE object
    """
    if xmin is None:
        xmin = kde.dataset.min()
    if xmax is None:
        xmax = kde.dataset.max()
    x = np.linspace(xmin,xmax,npts)
    return conf_interval(x,kde(x),shortest=shortest,conf=conf,
                         conftol=conftol,return_max=return_max)


def qstd(x,quant=0.05,top=False,bottom=False):
    """returns std, ignoring outer 'quant' pctiles
    """
    s = np.sort(x)
    n = np.size(x)
    lo = s[int(n*quant)]
    hi = s[int(n*(1-quant))]
    if top:
        w = np.where(x>=lo)
    elif bottom:
        w = np.where(x<=hi)
    else:
        w = np.where((x>=lo)&(x<=hi))
    return np.std(x[w])


def conf_interval(x,L,conf=0.683,shortest=True,
                  conftol=0.001,return_max=False):
    """
    Returns desired 1-d confidence interval for provided x, L[PDF]
    
    """
    cum = np.cumsum(L)
    cdf = cum/cum.max()
    if shortest:
        maxind = L.argmax()
        if maxind==0:   #hack alert
            maxind = 1
        if maxind==len(L)-1:
            maxind = len(L)-2
        Lval = L[maxind]

        lox = x[0:maxind]
        loL = L[0:maxind]
        locdf = cdf[0:maxind]
        hix = x[maxind:]
        hiL = L[maxind:]
        hicdf = cdf[maxind:]

        dp = 0
        s = -1
        dL = Lval
        switch = False
        last = 0
        while np.absolute(dp-conf) > conftol:
            Lval += s*dL
            if maxind==0:
                loind = 0
            else:
                loind = (np.absolute(loL - Lval)).argmin()
            if maxind==len(L)-1:
                hiind = -1
            else:
                hiind = (np.absolute(hiL - Lval)).argmin()

            dp = hicdf[hiind]-locdf[loind]
            lo = lox[loind]
            hi = hix[hiind]
            if dp == last:
                break
            last = dp
            cond = dp > conf
            if cond ^ switch:
                dL /= 2.
                s *= -1
                switch = not switch

    else:
        alpha = (1-conf)/2.
        lo = x[np.absolute(cdf-alpha).argmin()]
        hi = x[(np.absolute(cdf-(1-(alpha)))).argmin()]
        
    if return_max:
        xmaxL = x[L.argmax()]
        return xmaxL,lo,hi
    else:
        return (lo,hi)
