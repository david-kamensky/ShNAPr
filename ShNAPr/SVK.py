"""
The ``SVK`` module:
-------------------
Module for Kirchhoff--Love shell analysis using the St. Venant--Kirchhoff
(SVK) material model, with analytical through-thickness integration.  
The formulation is taken from

https://doi.org/10.1016/j.cma.2009.08.013
"""

from ShNAPr.kinematics import *

# TODO: Generalize this to take an arbitrary material matrix, and have a
# separate function to generate the isotropic one.

def surfaceEnergyDensitySVK(spline,X,x,E,nu,h_th,membrane=False,
                            membranePrestress=Constant(((0,0),(0,0))),
                            bendingPrestress=Constant(((0,0),(0,0)))):
    """
    Elastic energy per unit reference surface area for a shell with 
    reference configuration ``X``, deformed configuration ``x``, Young's
    modulus ``E``, Poisson's ratio ``nu``, and thickness ``h_th``.  
    Optionally, bending stiffness can be deactivated, by setting
    ``membrane`` to ``True``.  Membrane and bending prestresses
    may also be provided, in the local Cartesian coordinate basis obtained by
    Gram--Schmidt from the canonical curvilinear basis.
    """
    A0,A1,A2,_,A,B = surfaceGeometry(spline,X)
    a0,a1,a2,_,a,b = surfaceGeometry(spline,x)
    epsilon = 0.5*(a - A)
    kappa = B - b
    epsilonBar = covariantRank2TensorToCartesian2D(epsilon,A,A0,A1)
    kappaBar = covariantRank2TensorToCartesian2D(kappa,A,A0,A1)
    D = (E/(1.0 - nu*nu))*as_matrix([[1.0,  nu,   0.0         ],
                                     [nu,   1.0,  0.0         ],
                                     [0.0,  0.0,  0.5*(1.0-nu)]])
    nBar = h_th*D*voigt2D(epsilonBar)
    mBar = (h_th**3)*D*voigt2D(kappaBar)/12.0
    
    # DO NOT add prestresses directly to nBar and mBar, then plug them into
    # the standard formula for energy.  The resulting prestress will be off
    # by a factor of two.
        
    Wint = inner(voigt2D(epsilonBar),0.5*nBar
                 + voigt2D(membranePrestress,strain=False))
    if(not membrane):
        Wint += inner(voigt2D(kappaBar),0.5*mBar
                      + voigt2D(bendingPrestress,strain=False))
    return Wint

