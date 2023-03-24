"From Descoteaux to Tournier bases"
import numpy as np
def dimension(order):
    r"""
    Returns the dimension, :math:`R`, of the real, antipodally symmetric
    spherical harmonics basis for a given truncation order.
    Parameters
    ----------
    order : int
        The trunction order.
    Returns
    -------
    R : int
        The dimension of the truncated spherical harmonics basis.
    """
    return ((order + 1) * (order + 2)) // 2

def sh_degree(j):
    """
    Returns the degree, ``l``, of the spherical harmonic associated to index
    ``j``.
    Parameters
    ----------
    j : int
        The flattened index of the spherical harmonic.
    Returns
    -------
    l : int
        The associated even degree.
    """
    l = 0
    while dimension(l) - 1 < j:
        l += 2
    return l

def sh_order(j):
    """
    Returns the order, ``m``, of the spherical harmonic associated to index
    ``j``.
    Parameters
    ----------
    j : int
        The flattened index of the spherical harmonic.
    Returns
    -------
    m : int
        The associated order.
    """
    l = sh_degree(j)
    return j + l + 1 - dimension(l)

def convert_to_mrtrix(order):
    """
    Returns the linear matrix used to convert coefficients into the mrtrix
    convention for spherical harmonics.
    Parameters
    ----------
    order : int
    Returns
    -------
    conversion_matrix : array-like, shape (dim_sh, dim_sh)
    """
    dim_sh = dimension(order)
    conversion_matrix = np.zeros((dim_sh, dim_sh))
    for j in range(dim_sh):
        l = sh_degree(j)
        m = sh_order(j)
        if m == 0:
            conversion_matrix[j, j] = 1
        else:
            conversion_matrix[j, j - 2*m] = np.sqrt(2)
    return conversion_matrix
