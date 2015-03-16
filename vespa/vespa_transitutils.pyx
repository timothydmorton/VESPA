import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

FOO = "Hi" #for weird __init__ bug purposes

DTYPE = np.float
ctypedef np.float_t DTYPE_t
cdef DTYPE_t pi = 3.1415926535897932384626433832795

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double tanh(double)
    double sqrt(double)
    double atan2(double,double)
    double acos(double)
    double abs(double)
    double log(double)
    double ceil(double)
    
@cython.boundscheck(False)
def bindata(np.ndarray[DTYPE_t] ts, np.ndarray[DTYPE_t] fs, DTYPE_t dt):
    """Requires ts to be ordered
    """
    cdef long npts = len(ts)
    cdef DTYPE_t trange = ts[npts-1]-ts[0]
    cdef long nbins = int(ceil(trange/dt))

    cdef np.ndarray[DTYPE_t] tbin = np.empty(nbins,dtype=float)
    cdef np.ndarray[DTYPE_t] fbin = np.empty(nbins,dtype=float)
    cdef np.ndarray[DTYPE_t] ubin = np.empty(nbins,dtype=float)

    cdef unsigned int i
    cdef unsigned int j = 0
    cdef DTYPE_t lo = ts[0]
    cdef DTYPE_t hi = lo+dt
    cdef DTYPE_t fsum = 0
    cdef DTYPE_t fsumsq = 0
    cdef DTYPE_t tsum = 0
    cdef long count = 0
    for i in range(npts):
        if ts[i] > hi and count > 0:
            tbin[j] = tsum/count
            fbin[j] = fsum/count
            ubin[j] = sqrt(fsumsq/count - (fsum/count)*(fsum/count)) #std in bin
            lo = hi
            hi = lo+dt
            count = 0
            fsum=0
            fsumsq=0
            tsum=0
            j += 1
        else:
            tsum += ts[i]
            fsum += fs[i]
            fsumsq += fs[i]*fs[i]
            count += 1
        if i==npts-1:
            if count==0:
                tbin[j] = tbin[j-1]
                fbin[j] = fbin[j-1]
                ubin[j] = ubin[j-1]
            else:
                tbin[j] = tsum/count
                fbin[j] = fsum/count
                ubin[j] = sqrt(fsumsq/count - (fsum/count)*(fsum/count)) #std in bin
                
    return tbin,fbin,ubin
            

@cython.boundscheck(False)
def find_eclipse(np.ndarray[DTYPE_t] Es, DTYPE_t b,
                 DTYPE_t aR, DTYPE_t ecc, DTYPE_t w, DTYPE_t width, bool sec):
    cdef long npts = len(Es)
    w = w*pi/180. # degrees to radians
    cdef DTYPE_t inc
    if sec:
        inc = acos(b/aR*(1-ecc*sin(w))/(1-ecc*ecc))
    else:
        inc = acos(b/aR*(1+ecc*sin(w))/(1-ecc*ecc))
    cdef DTYPE_t nu,r,X,Y
    cdef DTYPE_t xmin = 10000.
    cdef bool on_rightside
    cdef np.ndarray[DTYPE_t] zs = np.empty(npts,dtype=float)
    cdef np.ndarray[long] in_eclipse = np.empty(npts,dtype=np.int) #how to make boolean array?
    cdef unsigned int i
    for i in range(npts):
        nu = 2 * atan2(sqrt(1+ecc)*sin(Es[i]/2),sqrt(1-ecc)*cos(Es[i]/2)) #true anomaly
        r = aR*(1-ecc*ecc)/(1+ecc*cos(nu))  #secondary distance from primary in units of R1
        X = -r*cos(w + nu)    
        Y = -r*sin(w + nu)*cos(inc)
        zs[i] = sqrt(X*X + Y*Y)
        if sec:
            on_rightside = (sin(nu + w) < 0)
        else:
            on_rightside = (sin(nu + w) >= 0)
        in_eclipse[i] = (on_rightside)*(zs[i] < width)
    return zs,in_eclipse
        
@cython.boundscheck(False)
def traptransit(np.ndarray[DTYPE_t] ts, np.ndarray[DTYPE_t] pars):
    """pars = [T,delta,T_over_tau,tc]"""
    cdef DTYPE_t t1 = pars[3] - pars[0]/2.
    cdef DTYPE_t t2 = pars[3] - pars[0]/2. + pars[0]/pars[2]
    cdef DTYPE_t t3 = pars[3] + pars[0]/2. - pars[0]/pars[2]
    cdef DTYPE_t t4 = pars[3] + pars[0]/2.
    cdef long npts = len(ts)
    cdef np.ndarray[DTYPE_t] fs = np.empty(npts,dtype=float)
    cdef unsigned int i
    for i in range(npts):
        if (ts[i]  > t1) and (ts[i] < t2):
            fs[i] = 1-pars[1]*pars[2]/pars[0]*(ts[i] - t1)
        elif (ts[i] > t2) and (ts[i] < t3):
            fs[i] = 1-pars[1]
        elif (ts[i] > t3) and (ts[i] < t4):
            fs[i] = 1-pars[1] + pars[1]*pars[2]/pars[0]*(ts[i]-t3)
        else:
            fs[i] = 1
    return fs

@cython.boundscheck(False)
def traptransit_resid( np.ndarray[DTYPE_t] pars, np.ndarray[DTYPE_t] ts, np.ndarray[DTYPE_t] fs):
    cdef long npts = len(ts)
    cdef np.ndarray[DTYPE_t] resid = np.empty(npts,dtype=float)
    cdef np.ndarray[DTYPE_t] fmod = traptransit(ts,pars)
    cdef unsigned int i
    for i in range(npts):
        resid[i] = fmod[i]-fs[i]
    return resid

@cython.boundscheck(False)
def traptransit_lhood(np.ndarray[DTYPE_t] pars, np.ndarray[DTYPE_t] ts, np.ndarray[DTYPE_t] fs, 
                      np.ndarray[DTYPE_t] sigs):
    cdef long npts = len(ts)
    cdef DTYPE_t lhood = 0
    cdef np.ndarray[DTYPE_t] fmod = traptransit(ts,pars)
    cdef unsigned int i
    for i in range(npts):
        lhood += -0.5*(fs[i]-fmod[i])*(fs[i]-fmod[i])/(sigs[i]*sigs[i])
    return lhood
    


@cython.boundscheck(False)
def protopapas(np.ndarray[DTYPE_t] ts, np.ndarray[DTYPE_t] pars):
    """pars: eta,theta,c,tau"""
    cdef DTYPE_t T = 10000.
    cdef long npts = len(ts)
    cdef np.ndarray[DTYPE_t] fs = np.empty(npts,dtype=float)
    cdef DTYPE_t tp = 0.
    cdef unsigned int i
    for i in range(npts):
        tp = T*sin(pi*(ts[i]-pars[3])/T)/pi/pars[0]
        fs[i] = 1 - pars[1] + 0.5*pars[1]*(2 - tanh(pars[2]*(tp+0.5)) + tanh(pars[2]*(tp-0.5)))
    return fs

@cython.boundscheck(False)
def protopapas_resid(np.ndarray[DTYPE_t] pars, np.ndarray[DTYPE_t] ts, np.ndarray[DTYPE_t] fs):
    cdef long npts = len(ts)
    cdef np.ndarray[DTYPE_t] resid = np.empty(npts,dtype=float)
    cdef np.ndarray[DTYPE_t] fmod = protopapas(ts,pars)
    cdef unsigned int i
    for i in range(npts):
        resid[i] = fmod[i]-fs[i]
    return resid
     

#####Mandel/Agol utility functions below.  UNFINISHED; will not work.

## Computes Hasting's polynomial approximation for the complete
## elliptic integral of the first (ek) and second (kk) kind
#@cython.boundscheck(False)
#def ellke(DTYPE_t k):
#    DTYPE_t m1=1.-k**2
#    DTYPE_t logm1 = log(m1)

#    DTYPE_t a1=0.44325141463
#    DTYPE_t a2=0.06260601220
#    DTYPE_t a3=0.04757383546
#    DTYPE_t a4=0.01736506451
#    DTYPE_t b1=0.24998368310
#    DTYPE_t b2=0.09200180037
#    DTYPE_t b3=0.04069697526
#    DTYPE_t b4=0.00526449639
#    DTYPE_t ee1=1.+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
#    DTYPE_t ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*(-logm1)
#    DTYPE_t ek = ee1+ee2
        
#    DTYPE_t a0=1.38629436112
#    DTYPE_t a1=0.09666344259
#    DTYPE_t a2=0.03590092383
#    DTYPE_t a3=0.03742563713
#    DTYPE_t a4=0.01451196212
#    DTYPE_t b0=0.5
#    DTYPE_t b1=0.12498593597
#    DTYPE_t b2=0.06880248576
#    DTYPE_t b3=0.03328355346
#    DTYPE_t b4=0.00441787012
#    DTYPE_t ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
#    DTYPE_t ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*logm1
#    DTYPE_t kk = ek1-ek2
    
#    return [ek,kk]

## Computes the complete elliptical integral of the third kind using
## the algorithm of Bulirsch (1965):
#@cython.boundscheck(False)
#def ellpic_bulirsch(DTYPE_t n,DTYPE_t k):
#    DTYPE_t kc=sqrt(1.-k*k)
#    DTYPLE_t p=n+1.
#    if(p.min() < 0.):
#        print 'Negative p'
#    m0=1.; c=1.; p=sqrt(p); d=1./p; e=kc
#    while 1:
#        f = c; c = d/p+c; g = e/p; d = 2.*(f*g+d)
#        p = g + p; g = m0; m0 = kc + m0
#        if (absolute(1.-kc/g)).max() > 1.e-8:
#            kc = 2*sqrt(e); e=kc*m0
#        else:
#            return 0.5*pi*(c*m0+d)/(m0*(m0+p))
