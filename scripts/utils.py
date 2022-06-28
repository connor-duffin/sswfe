import fenics as fe

from scipy.sparse import csr_matrix
from petsc4py.PETSc import Mat


def dolfin_to_csr(A):
    """
    Convert assembled matrix to scipy CSR.

    Parameters
    ----------
    A : fenics.Matrix or PETSc4py.Mat
        Sparse matrix to convert to scipy csr matrix.
    """
    if type(A) != Mat:
        mat = fe.as_backend_type(A).mat()
    else:
        mat = A
    csr = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    return csr
