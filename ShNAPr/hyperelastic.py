"""
The ``hyperelastic`` module:
----------------------------
Module for Kirchhoff--Love shell analysis using incompressible hyperelastic
models, as often found, e.g., in biomedical problems.  For the theory, see

https://doi.org/10.1016/j.cma.2015.03.010
"""

from ShNAPr.kinematics import *

def incompressiblePressureKL(psi_el,E):
    """
    Compute the pressure Lagrange multiplier for the incompressibility
    constraint, given Green--Lagrange strain ``E`` in Cartesian coordinates
    and hyperelastic energy ``psi_el``, which is a function of Cartesian
    Green--Lagrange strain.
    """
    E = variable(E)
    dpsi_el_dC = 0.5*diff(psi_el(E),E)
    C22 = 2.0*E[2,2] + 1.0
    return 2.0*dpsi_el_dC[2,2]*C22

def incompressiblePotentialKL(spline,X,x,psi_el):
    """
    Return the total strain energy density, as a Python function of the
    through-thickness coordinate ``xi2``, given the elastic potential
    ``psi_el``, as a Python function of 3D Cartesian Green--Lagrange strain,
    the ``spline`` defining the parameter and function spaces, and the 
    reference and deformed configurations, ``X`` and ``x``.
    """
    A0,A1,A2,deriv_A2,A,B = surfaceGeometry(spline,X)
    a0,a1,a2,deriv_a2,a,b = surfaceGeometry(spline,x)
    def closure(xi2):
        G = metricKL(A,B,xi2)
        g = metricKL(a,b,xi2)
        E_flat = 0.5*(g - G)
        G0,G1 = curvilinearBasisKL(A0,A1,deriv_A2,xi2)
        E_2D = covariantRank2TensorToCartesian2D(E_flat,G,G0,G1)
        C_2D = 2.0*E_2D + Identity(2)
        C22 = 1.0/det(C_2D)
        E22 = 0.5*(C22-1.0)
        E = as_tensor([[E_2D[0,0], E_2D[0,1], 0.0],
                       [E_2D[1,0], E_2D[1,1], 0.0],
                       [0.0,       0.0,       E22]])
        C = 2.0*E + Identity(3)
        J = sqrt(det(C))
        p = incompressiblePressureKL(psi_el,E)
        return psi_el(E) - p*(J-1.0)
    return closure
