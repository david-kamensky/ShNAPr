"""
The ``contact`` module:
-----------------------
This module provides functionality for general frictionless self-contact 
of shell structures, using the nonlocal formulation given in

  https://doi.org/10.1016/j.cma.2017.11.007

Note that an objective, angular-momentum-conserving frictional extension 
for shell structures is outlined in Remark 7 of

  https://doi.org/10.1007%2Fs42102-019-00012-y

although it has never been implemented.  Alternatively, it would be quite
easy to implement the "naive" fricitional model in the cited reference.
"""

from tIGAr import *
from numpy import zeros
from scipy.spatial import cKDTree
from numpy.linalg import norm as npNorm
from numpy import outer as npOuter
from numpy import identity as npIdentity

# Shells only exist in three space dimensions.
d = 3

# Potentially-fragile assumption on the ordering of DoFs in mixed-element
# function spaces:
def nodeToDof(node,direction):
    return d*node + direction

# Overkill preallocation for the contact tangent matrix; if a node interacts
# with too many other nodes, this could be exceeded, and the contact force
# assembly will slow down drastically.
PREALLOC = 500

# For rapid prototyping, one can numerically take the derivative of
# $\phi'$ needed for the LHS of the formulation.  This is the hard-coded
# epsilon used for a finite difference.
PHI_EPS = 1e-10

class ShellContactContext:
    """
    Because there is some one-time initialization associated with the contact
    method, it makes sense to hold a persistent state in an object.
    """
    def __init__(self,spline,R_self,r_max,
                 phiPrime,phiDoublePrime=None):
        """
        ``spline`` is the ``tIGAr`` ``ExtractedSpline`` object defining the
        shell structure.  ``R_self`` is the radius defining the reference-
        configuration neighborhood with which each point does not interact
        through contact forces.  ``r_max`` is the maximum range used to
        identify potentially-contacting points in the current configuration;
        this should ideally be the radius of the support of ``phiPrime``,
        which defines the magnitude of central contact forces as a function
        of distance between points.  It's derivative, ``phiDoublePrime``
        is ideally passed as well, but, if it is omitted, a finite-difference
        approximation is taken.  The functions ``phiPrime`` and
        ``phiDoublePrime`` should be Python functions, each taking a single
        real-valued argument.  
        """
        if(phiDoublePrime==None):
            # Do not use centered or backward differences, because
            # arguments to $\phi$ are assumed positive.
            phiDoublePrime = lambda r : (phiPrime(r+PHI_EPS)
                                         - phiPrime(r))/PHI_EPS
        self.spline = spline
        self.phiPrime = phiPrime
        self.phiDoublePrime = phiDoublePrime
        self.R_self = R_self
        self.r_max = r_max
        
        # Potentially-fragile assumption: that there is a correspondence
        # in DoF order between the scalar space used for each component of
        # the control mapping and the mixed space used for the displacement.  
        self.nNodes = self.spline.cpFuncs[0].vector().get_local().size
        self.nodeX = zeros((self.nNodes,d))
        # (Could be optimized for numba, but not worthwhile for one-time
        # initialization step.)
        for i in range(0,self.nNodes):
            wi = self.spline.cpFuncs[d].vector().get_local()[i]
            for j in range(0,d):
                Xj_hom = self.spline.cpFuncs[j].vector().get_local()[i]
                self.nodeX[i,j] = Xj_hom/wi

        # Using quadrature points coincident with the FE nodes of the
        # extracted representation of the spline significantly simplifies
        # the assembly process.
        W = assemble(inner(Constant(d*(1.0,)),
                           TestFunction(spline.V))*spline.dx)

        # Unfortunately, mass lumping with super-linear Lagrange FEs on
        # simplices is not robust, and leads to some negative weights.
        # The following mass-conservative smoothing procedure improves
        # performance.
        if(self.spline.mesh.ufl_cell()==triangle):
            u = TrialFunction(self.spline.V)
            v = TestFunction(self.spline.V)
            w = Function(self.spline.V)
            w.vector()[:] = W
            smoothL = Constant(2.0)*CellDiameter(self.spline.mesh)
            smoothRes = inner(u-w,v)*dx + (smoothL**2)*inner(grad(u),
                                                             grad(v))*dx
            w_smoothed = Function(self.spline.V)
            solve(lhs(smoothRes)==rhs(smoothRes),w_smoothed)
            W = w_smoothed.vector()

        quadWeightsTemp = W.get_local()
        self.quadWeights = zeros(self.nNodes)
        for i in range(0,self.nNodes):
            self.quadWeights[i] = quadWeightsTemp[nodeToDof(i,0)]            

    def evalFunction(self,vFunc):
        """
        Obtain a ``self.nNodes``-by-``d`` array containing physical values of
        the homogeneous function ``vFunc`` from space ``self.spline.V`` 
        evaluated at FE nodes.
        """
        vFlat = vFunc.vector().get_local()
        v = vFlat.reshape((-1,d))
        # Divide nodal velocity/displacement through by FE nodal weights.
        for i in range(0,self.nNodes):
            wi = self.spline.cpFuncs[d].vector().get_local()[i]
            for j in range(0,d):
                v[i,j] /= wi
        return v
    
    def assembleContact(self,dispFunc):
        """
        Return FE stiffness matrix and load vector contributions associated 
        with contact forces, based on an FE displacement ``dispFunc``.  
        """
        
        # Establish tensors to accumulate contact contributions.
        F = assemble(inner(Constant(d*(0.0,)),
                           TestFunction(self.spline.V))
                     *dx(metadata={"quadrature_degree":0}),
                     finalize_tensor=False)
        Fv = as_backend_type(F).vec()
        # Using fact that
        KPETSc = PETSc.Mat(self.spline.comm)
        KPETSc.createAIJ([[d*self.nNodes,None],[None,d*self.nNodes]],
                         comm=self.spline.comm)
        KPETSc.setPreallocationNNZ([PREALLOC,PREALLOC])
        KPETSc.setUp()
        K = PETScMatrix(KPETSc)
        Km = as_backend_type(K).mat()
        ADD_MODE = PETSc.InsertMode.ADD
        
        # Ideally, we would first examine the set of pairs
        # returned by the cKDTree query, then allocate based
        # on that, rather than the present approach of
        # preallocating a large number of nonzeros and hoping
        # for the best.  
        Km.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,False)
       
        ## Compute deformed positions of nodes in physical space.
        #dispFlat = dispFunc.vector().get_local()
        splineWeightVec = self.spline.cpFuncs[d].vector().get_local()
        nodex = self.nodeX + self.evalFunction(dispFunc)

        tree = cKDTree(nodex)
        pairs = tree.query_pairs(self.r_max,output_type='ndarray')

        # Because the ndarray output from the scipy cKDTree maps onto a C++
        # type, this loop could likely be optimized further by placing it in
        # a JIT-compiled C++ extension module.  (For this, string
        # representations of phiPrime and phiDoublePrime would be needed.)
        for pair in pairs:

            node1 = pair[0]
            node2 = pair[1]

            # Positions of the two nodal quadrature points in the reference
            # configuration:
            X1 = self.nodeX[node1,:]
            X2 = self.nodeX[node2,:]
            R12 = npNorm(X2-X1)

            # Do not add contact forces between points that are close in the
            # reference configuration.  (Otherwise, the entire structure 
            # would expand, trying to get away from itself.)
            if(R12 > self.R_self):

                # Positions of nodes in the current configuration:
                x1 = nodex[node1,:]
                x2 = nodex[node2,:]

                # Force computation: see (24) from original reference.
                r12vec = x2-x1
                r12 = npNorm(r12vec)
                r12hat = r12vec/r12
                r_otimes_r = npOuter(r12hat,r12hat)
                I = npIdentity(d)
                C = self.quadWeights[node1]*self.quadWeights[node2]
                f12 = C*self.phiPrime(r12)*r12hat

                # Nodal FE spline (not quadrature) weights:
                w1 = splineWeightVec[node1]
                w2 = splineWeightVec[node2]

                # Add equal-and-opposite forces to the RHS vector.
                for direction in range(0,d):
                    dof1 = nodeToDof(node1,direction)
                    dof2 = nodeToDof(node2,direction)
                    Fv.setValue(dof1,-f12[direction]/w1,addv=ADD_MODE)
                    Fv.setValue(dof2,f12[direction]/w2,addv=ADD_MODE)
                    # (Weights are involved here because the FE test function
                    # is in homogeneous representation.)

                # Tangent computation: see (25)--(26) from original
                # reference.  
                k12_tensor = C*(self.phiDoublePrime(r12)*r_otimes_r 
                                + (self.phiPrime(r12)/r12)*(I-r_otimes_r))

                # Add tangent contributions to the LHS matrix.
                for d1 in range(0,d):
                    for d2 in range(0,d):
                        n1dof1 = nodeToDof(node1,d1)
                        n1dof2 = nodeToDof(node1,d2)
                        n2dof1 = nodeToDof(node2,d1)
                        n2dof2 = nodeToDof(node2,d2)
                        k12 = k12_tensor[d1,d2]

                        # 11 contribution:
                        Km.setValue(n1dof1,n1dof2,k12/(w1*w1),addv=ADD_MODE)
                        # 22 contribution:
                        Km.setValue(n2dof1,n2dof2,k12/(w2*w2),addv=ADD_MODE)
                        # Off-diagonal contributions:
                        Km.setValue(n1dof1,n2dof2,-k12/(w1*w2),addv=ADD_MODE)
                        Km.setValue(n2dof1,n1dof2,-k12/(w1*w2),addv=ADD_MODE)
                        # (Weights are involved here because FE test and 
                        # trial space basis functions are in homogeneous
                        # representation.)
        Fv.assemble()
        Km.assemble()
        return K, F

class ShellContactNonlinearProblem(ExtractedNonlinearProblem):
    """
    Class encapsulating a nonlinear problem with an isogeometric shell 
    formulation and shell contact.  
    """
    def __init__(self,contactContext,residual,tangent,solution,**kwargs):
        self.contactContext = contactContext
        self.spline = self.contactContext.spline
        super(ShellContactNonlinearProblem, self)\
            .__init__(self.spline,residual,tangent,solution,**kwargs)
    # Override methods from NonlinearProblem to perform extraction and
    # include contact forces:
    def form(self,A,P,B,x):
        self.solution.vector()[:] = self.spline.M*x
        self.Kc,self.Fc = self.contactContext.assembleContact(self.solution)
    def F(self,b,x):
        b[:] = self.spline.extractVector(assemble(self.residual)
                                         + self.Fc)
        return b
    def J(self,A,x):
        M = self.spline.extractMatrix(as_backend_type(assemble(self.tangent))
                                      + self.Kc)
        Mm = as_backend_type(M).mat()
        A.mat().setSizes(Mm.getSizes())
        A.mat().setUp()
        Mm.copy(result=A.mat())
        return A
