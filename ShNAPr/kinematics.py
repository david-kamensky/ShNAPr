"""
The ``kinematics`` module:
--------------------------
Base module providing the Kirchhoff--Love kinematics needed for various
different constitutive models.  For an overview of the theory, refer to

http://mediatum.ub.tum.de/doc/1002634/464162.pdf
"""

from tIGAr import *

def unit(v):
    """
    Normalize the vector ``v``.
    """
    return v/sqrt(inner(v,v))

def curvilinearBasisKL(a0,a1,deriv_a2,xi2):
    """
    Generate a curvilinear basis from midsurface data at the 
    through-thickness coordinate ``xi2``.  Argument names follow the
    convention from the return value of ``surfaceGeometry``.  
    """
    g0 = a0 + xi2*deriv_a2[:,0]
    g1 = a1 + xi2*deriv_a2[:,1]
    return g0,g1

def metricKL(a,b,xi2):
    """
    Generate a metric tensor at through-thickness coordinate ``xi2``, 
    based on the midsurface metric ``a`` and curvature ``b``, assuming
    Kichhoff--Love kinematics.
    """
    return a - 2.0*xi2*b

def orthonormalize2D(a0,a1):
    """
    2D Gram--Schmidt procedure.
    """
    e0 = unit(a0)
    e1 = unit(a1 - e0*inner(a1,e0))
    return e0, e1

def surfaceGeometry(spline,x):
    """
    Obtain surface geometry data from a configuration of``spline``,
    given by `x`.  Returns a tuple with:  Curvilinear basis vectors
    ``a0``, ``a1``, and ``a2``, the parametric derivatives ``deriv_a2`` of
    ``a2``, the metric tensor ``a``, and the curvature tensor ``b``.
    """
    # Covariant basis vectors
    dxdxi = spline.parametricGrad(x)
    a0 = as_vector([dxdxi[0,0],dxdxi[1,0],dxdxi[2,0]])
    a1 = as_vector([dxdxi[0,1],dxdxi[1,1],dxdxi[2,1]])
    a2 = unit(cross(a0,a1))

    # Midsurface metric tensor
    a = as_matrix(((inner(a0,a0),inner(a0,a1)),\
                   (inner(a1,a0),inner(a1,a1))))
    # Curvature
    deriv_a2 = spline.parametricGrad(a2)
    b = -as_matrix(((inner(a0,deriv_a2[:,0]),inner(a0,deriv_a2[:,1])),
                    (inner(a1,deriv_a2[:,0]),inner(a1,deriv_a2[:,1]))))

    return (a0,a1,a2,deriv_a2,a,b)

def covariantRank2TensorToCartesian2D(T,a,a0,a1):
    """
    Convert a curvilinear tensor of type ``$\\binom{0}{2}$`` in a 2D tangent
    space to a local Cartesian basis.  The tensor is ``T``, the metric 
    is ``a``, and the curvilinear basis vectors are ``a0`` and ``a1``.  

    NOTE:  With approximate kinematics, e.g., Kirchhoff--Love, the metric may
    not follow from the curvilinear basis vectors, so it is not redundant 
    to pass both.
    """
    
    # raise indices on curvilinear basis
    ac = inv(a)
    a0c = ac[0,0]*a0 + ac[0,1]*a1
    a1c = ac[1,0]*a0 + ac[1,1]*a1

    e0,e1 = orthonormalize2D(a0,a1)

    ea = as_matrix(((inner(e0,a0c),inner(e0,a1c)),
                    (inner(e1,a0c),inner(e1,a1c))))
    ae = ea.T

    return ea*T*ae

def voigt2D(T,strain=True):
    """
    Convert a 2D symmetric rank-2 tensor ``T`` to Voigt notation.  If
    ``strain`` is true (the default), then the convention for strains is
    followed, where the off-diagonal component is doubled.
    """
    if(strain):
        fac = 2.0
    else:
        fac = 1.0
    return as_vector([T[0,0],T[1,1],fac*T[0,1]])

def invVoigt2D(v,strain=True):
    """
    Inverse of ``voigt2D``.
    """
    if(strain):
        fac = 2.0
    else:
        fac = 1.0
    return as_tensor([[v[0]     , v[2]/fac],
                      [v[2]/fac , v[1]    ]])

class throughThicknessMeasure:
    """
    Class to represent a local integration through the thickness of a shell.
    The ``__rmul__`` method is overloaded for an instance ``dxi2`` to be
    used like ``volumeIntegral = volumeIntegrand*dxi2*spline.dx``, where
    ``volumeIntegrand`` is a python function taking a single parameter,
    ``xi2``.
    """
    def __init__(self,nPoints,h):
        """
        Integration uses a quadrature rule with ``nPoints`` points, and assumes
        a thickness ``h``.
        """
        self.nPoints = nPoints
        self.h = h
        self.xi2, self.w = getQuadRuleInterval(nPoints,h)

    def __rmul__(self,integrand):
        """
        Given an ``integrand`` that is a Python function taking a single
        ``float`` parameter with a valid range of ``-self.h/2`` to 
        ``self.h/2``, return the (numerical) through-thickness integral.
        """
        integral = 0.0
        for i in range(0,self.nPoints):
            integral += integrand(self.xi2[i])*self.w[i]
        return integral
