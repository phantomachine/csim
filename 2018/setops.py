import numpy as np
import itertools

#---------- SET-DIFFERENCE OF ARRAYS -------------------------------------
def setdiff(a1, a2):
    """Simpler version of Matlab/R's SETDIFF for 2D arrays only """
    a1 = a1.ravel().reshape(-1,a1.shape[-1])
    a2 = a2.ravel().reshape(-1,a1.shape[-1])
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[-1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[-1])
    ad = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[-1])
    index_bool = np.in1d(a1_rows, a2_rows, invert=True)
    index = np.arange(index_bool.size)[index_bool]
    return ad, index, index_bool

#---------- INTERSECTION OF ARRAYS --------------------------------------
def intersect(a1, a2):
    """Simpler version of Matlab/R's INTERSECT for 2D arrays only """
    a1 = a1.ravel().reshape(-1,a1.shape[-1])
    a2 = a2.ravel().reshape(-1,a1.shape[-1])
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[-1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[-1])
    ax=np.intersect1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[-1])
    index_bool = np.in1d(a1_rows, a2_rows, invert=False)
    index = np.arange(index_bool.size)[index_bool]
    return ax, index, index_bool

#------------- PYTHON LIST or NUMPY ARRAYS: Flatten vs. Reconstruct -------

def flatten(nl):
    """
    To flatten Python List of lists / numpy arrays (1 level). (See also reverse operation in RECONSTRUCT() below.)
    Usage: L_flat,l1,l2 = flatten(L)
    Source: http://stackoverflow.com/questions/27982432/flattening-and-unflattening-a-nested-list-of-numpy-arrays
    """
    l1 = len(nl)
    nl = list(itertools.chain.from_iterable(nl))
    return nl,l1

def reconstruct(nl, l1):
    """
    To reconstruct Python List of lists / numpy arrays. Inverse operation of FLATTEN() above.
    Usage: L_reconstructed = reconstruct2(L_flat,l1)
    Source: http://stackoverflow.com/questions/27982432/flattening-and-unflattening-a-nested-list-of-numpy-arrays
    """
    return np.split(nl,np.cumsum(l1))[:-1]

def flatten2(nl):
    """
    To flatten Python List of lists / numpy arrays (2 levels). (See also reverse operation in RECONSTRUCT() below.)
    Usage: L_flat,l1,l2 = flatten2(L)
    Source: http://stackoverflow.com/questions/27982432/flattening-and-unflattening-a-nested-list-of-numpy-arrays
    """
    l1 = [len(s) for s in itertools.chain.from_iterable(nl)]
    l2 = [len(s) for s in nl]

    nl = list(itertools.chain.from_iterable(itertools.chain.from_iterable(nl)))

    return nl,l1,l2

def reconstruct2(nl, l1, l2):
    """
    To reconstruct Python List of lists / numpy arrays. Inverse operation of FLATTEN() above.
    Usage: L_reconstructed = reconstruct2(L_flat,l1,l2)
    Source: http://stackoverflow.com/questions/27982432/flattening-and-unflattening-a-nested-list-of-numpy-arrays
    """
    return np.split(np.split(nl,np.cumsum(l1)),np.cumsum(l2))[:-1]
