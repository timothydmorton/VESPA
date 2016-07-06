try:
    import numpy as np
    import hashlib

    from hashlib import sha1

    from numpy import all, array, uint8
except ImportError:
    np, hashlib, sha1 = (None, None, None)
    all, array, uint8 = (None, None, None)

class hashable(object):
    r'''Hashable wrapper for ndarray objects.

        Instances of ndarray are not hashable, meaning they cannot be added to
        sets, nor used as keys in dictionaries. This is by design - ndarray
        objects are mutable, and therefore cannot reliably implement the
        __hash__() method.

        The hashable class allows a way around this limitation. It implements
        the required methods for hashable objects in terms of an encapsulated
        ndarray object. This can be either a copied instance (which is safer)
        or the original object (which requires the user to be careful enough
        not to modify it).

        This class taken from `here <http://stackoverflow.com/questions/1939228/constructing-a-python-set-from-a-numpy-matrix/5173201#5173201>`_; edited only slightly.
    '''
    def __init__(self, wrapped, tight=False):
        r'''Creates a new hashable object encapsulating an ndarray.

            wrapped
                The wrapped ndarray.

            tight
                Optional. If True, a copy of the input ndaray is created.
                Defaults to False.
        '''
        self.__tight = tight
        self.__wrapped = array(wrapped) if tight else wrapped
        #self.__hash = int(sha1(wrapped.view(uint8)).hexdigest(), 16)
        self.__hash = int(sha1(np.ascontiguousarray(wrapped)).hexdigest(), 16)

    def __eq__(self, other):
        return all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        r'''Returns the encapsulated ndarray.

            If the wrapper is "tight", a copy of the encapsulated ndarray is
            returned. Otherwise, the encapsulated ndarray itself is returned.
        '''
        if self.__tight:
            return array(self.__wrapped)

        return self.__wrapped

def hasharray(arr):
    """
    Hashes array-like object (except DataFrame)
    """
    #return hash(hashlib.sha1(np.ascontiguousarray(arr)).hexdigest())
    return hash(hashable(np.array(arr)))

def hashdf(df):
    """hashes a pandas dataframe, forcing values to float
    """
    return hasharray(df.values.astype(float))

def hashcombine(*xs):
    """
    Combines multiple hashes using xor
    """
    k = 0
    for x in xs:
        k ^= hash(x)
    k ^= hash(xs)
    return k

def hashdict(d):
    """Hash a dictionary
    """
    k = 0
    for key,val in d.items():
        k ^= hash(key) ^ hash(val)
    return k
