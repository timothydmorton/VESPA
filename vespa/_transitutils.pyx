import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

FOO = "Hi" #for weird __init__ bug purposes

DTYPE = np.float
ctypedef np.float_t DTYPE_t
cdef DTYPE_t pi = 3.1415926535897932384626433832795
cdef DTYPE_t TAU = 2*pi

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double tanh(double)
    double sqrt(double)
    double atan2(double,double)
    double acos(double)
    double asin(double)
    double abs(double)
    double fabs(double)
    double log(double)
    double ceil(double)
    float INFINITY
    

cdef int KEPLER_MAX_ITER = 100
cdef DTYPE_t KEPLER_CONV_TOL = 1e-5

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_eccentric_anomaly(DTYPE_t mean_anomaly, DTYPE_t eccentricity):
    
    cdef DTYPE_t guess = mean_anomaly
    cdef unsigned int i = 0
    
    cdef DTYPE_t f
    cdef DTYPE_t f_prime
    cdef DTYPE_t newguess
    cdef DTYPE_t sguess

    for i in xrange(KEPLER_MAX_ITER):
        sguess = sin(guess)
        f = guess - eccentricity * sguess - mean_anomaly
        f_prime = 1 - eccentricity * cos(guess)
        newguess = guess - f/f_prime
        if fabs(newguess - guess) < KEPLER_CONV_TOL:
            return newguess
        guess = newguess

    return guess

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t true_anomaly(DTYPE_t M, DTYPE_t ecc):

    cdef DTYPE_t guess = M
    cdef unsigned int i = 0
    
    cdef DTYPE_t f
    cdef DTYPE_t f_prime
    cdef DTYPE_t E
    
    for i in xrange(KEPLER_MAX_ITER):
        f = guess - ecc * sin(guess) - M
        f_prime = 1 - ecc * cos(guess)
        E = guess - f/f_prime
        if fabs(E - guess) < KEPLER_CONV_TOL:
            break
        guess = E

    return 2 * atan2(sqrt(1 + ecc) * sin(E/2.),
                     sqrt(1 - ecc) * cos(E/2.))


@cython.boundscheck(False)
@cython.wraparound(False)
def zs_of_Ms(np.ndarray[DTYPE_t] Ms, DTYPE_t b, DTYPE_t aR, DTYPE_t ecc, 
             DTYPE_t w, bool sec=False):
    cdef long npts = len(Ms)

    zs = np.empty(npts, dtype=float)

    for i in xrange(npts):
        zs[i] = z_of_M(Ms[i], b, aR, ecc, w, sec)

    return zs

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t z_of_M(DTYPE_t M, DTYPE_t b, DTYPE_t aR, DTYPE_t ecc, DTYPE_t w, bool sec=False):
    
    cdef DTYPE_t inc
    if sec:
        inc = acos(b/aR * (1 - ecc*sin(w))/(1 - ecc*ecc))
    else:
        inc = acos(b/aR * (1 + ecc*sin(w))/(1 - ecc*ecc))

    cdef DTYPE_t nu = true_anomaly(M, ecc)

    if sec:
        if sin(nu + w) >= 0:
            return INFINITY
    else:
        if sin(nu + w) < 0:
            return INFINITY

    cdef DTYPE_t sin_i = sin(inc)
    cdef DTYPE_t sin_wf = sin(w + nu)

    cdef DTYPE_t r_sky = (aR*(1-ecc*ecc) / (1 + ecc*cos(nu)) *
                          sqrt(1 - sin_wf*sin_wf * sin_i*sin_i))

    return r_sky


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DTYPE_t angle_from_transit(DTYPE_t M, DTYPE_t ecc, DTYPE_t w):
    cdef DTYPE_t f = true_anomaly(M, ecc)
    cdef DTYPE_t f0 = pi/2 - w
    cdef DTYPE_t a = (f - f0) % TAU
    cdef DTYPE_t b = (f0 - f) % TAU
    if a < b:
        return a
    else:
        return b
    #return fabs(true_anomaly(M, ecc) - (pi/2 - w))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DTYPE_t angle_from_occultation(DTYPE_t M, DTYPE_t ecc, DTYPE_t w):
    cdef DTYPE_t f = true_anomaly(M, ecc)
    cdef DTYPE_t f0 = -pi/2 - w
    cdef DTYPE_t a = (f - f0) % TAU
    cdef DTYPE_t b = (f0 - f) % TAU
    if a < b:
        return a
    else:
        return b
    #return fabs(true_anomaly(M, ecc) - (-pi/2 - w))


@cython.boundscheck(False)
@cython.wraparound(False)
def transit_duration(DTYPE_t k, DTYPE_t P, DTYPE_t b, DTYPE_t aR, DTYPE_t ecc=0., 
                     DTYPE_t w=0, bool sec=False):
    """
    Duration from z=1 to z=1 (halfway b/w full and partial)
    """
    cdef DTYPE_t inc
    cdef DTYPE_t eccfactor

    cdef DTYPE_t eccsq = ecc*ecc
    cdef DTYPE_t esinw = ecc*sin(w)

    if sec:
        inc = acos(b/aR * (1 - esinw)/(1 - eccsq))
        eccfactor = sqrt(1 - eccsq) / (1 - esinw)
    else:
        inc = acos(b/aR * (1 + esinw)/(1 - eccsq))
        eccfactor = sqrt(1 - eccsq) / (1 + esinw)

    return P/pi * asin(1./aR * sqrt((1+k)*(1+k) - b*b) / sin(inc)) * eccfactor

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
    for i in xrange(npts):
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
    for i in xrange(npts):
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
    cdef long npts = len(ts)
    cdef np.ndarray[DTYPE_t] fs = np.empty(npts,dtype=float)
    cdef DTYPE_t t1, t2, t3, t4
    cdef unsigned int i

    if pars[2] < 2 or pars[0] <= 0:
        for i in xrange(npts):
            fs[i] = INFINITY
    else:
        t1 = pars[3] - pars[0]/2.
        t2 = pars[3] - pars[0]/2. + pars[0]/pars[2]
        t3 = pars[3] + pars[0]/2. - pars[0]/pars[2]
        t4 = pars[3] + pars[0]/2.
        for i in xrange(npts):
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
    cdef unsigned int i
    cdef np.ndarray[DTYPE_t] fmod = traptransit(ts,pars)
    for i in xrange(npts):
        resid[i] = fmod[i]-fs[i]
    return resid

@cython.boundscheck(False)
def traptransit_lhood(np.ndarray[DTYPE_t] pars, np.ndarray[DTYPE_t] ts, np.ndarray[DTYPE_t] fs, 
                      np.ndarray[DTYPE_t] sigs):
    cdef long npts = len(ts)
    cdef DTYPE_t lhood = 0
    cdef np.ndarray[DTYPE_t] fmod = traptransit(ts,pars)
    cdef unsigned int i
    for i in xrange(npts):
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
    for i in xrange(npts):
        tp = T*sin(pi*(ts[i]-pars[3])/T)/pi/pars[0]
        fs[i] = 1 - pars[1] + 0.5*pars[1]*(2 - tanh(pars[2]*(tp+0.5)) + tanh(pars[2]*(tp-0.5)))
    return fs

@cython.boundscheck(False)
def protopapas_resid(np.ndarray[DTYPE_t] pars, np.ndarray[DTYPE_t] ts, np.ndarray[DTYPE_t] fs):
    cdef long npts = len(ts)
    cdef np.ndarray[DTYPE_t] resid = np.empty(npts,dtype=float)
    cdef np.ndarray[DTYPE_t] fmod = protopapas(ts,pars)
    cdef unsigned int i
    for i in xrange(npts):
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
