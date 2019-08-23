"""
This demo illustrates the usage of ``ShNAPr.hyperelastic``.  The problem
considered here is the inflation of a neo-Hookean membrane.
"""

from tIGAr import *
from tIGAr.BSplines import *
from tIGAr.timeIntegration import *
from ShNAPr.kinematics import *
from ShNAPr.hyperelastic import *

# Use TSFC representation, due to complicated forms:
parameters["form_compiler"]["representation"] = "tsfc"
import sys
sys.setrecursionlimit(10000)

####### Preprocessing #######

if(mpirank==0):
    print("Generating extraction data...")

# Specify the number of elements in each direction.
NELu = 16
NELv = 16

# Specify degree in each direction.
degs = [2,2]

# Generate open knot vectors for each direction.
kvecs = [uniformKnots(degs[0],-1.0,1.0,NELu),
         uniformKnots(degs[1],-1.0,1.0,NELv)]

# Generate an explicit B-spline control mesh.  The argument extraDim allows
# for increasing the dimension of physical space beyond that of parametric
# space.  We want to model deformations in 3D, so one extra dimension is
# required of physical space.
controlMesh = ExplicitBSplineControlMesh(degs,kvecs,extraDim=1)

# Create a spline generator with three unknown fields for the displacement
# components.  
splineGenerator = EqualOrderSpline(3,controlMesh)

# Apply clamped boundary conditions to the displacement.  (Pinned BCs are in
# comments, but need more load steps and/or a smaller load to converge.)
scalarSpline = splineGenerator.getControlMesh().getScalarSpline()
for side in range(0,2):
    for direction in range(0,2):
        sideDofs = scalarSpline.getSideDofs(direction,side,
                                            ########################
                                            nLayers=2) # clamped BC
                                            #nLayers=1) # pinned BC
                                            ########################
        for i in range(0,3):
            splineGenerator.addZeroDofs(i,sideDofs)

####### Analysis #######

if(mpirank==0):
    print("Creating extracted spline...")

# Quadrature degree for the analysis:
QUAD_DEG = 4

# Generate the extracted representation of the spline.
spline = ExtractedSpline(splineGenerator,QUAD_DEG)

if(mpirank==0):
    print("Starting analysis...")
    
# Unknown midsurface displacement
y_hom = Function(spline.V) # in homogeneous representation
y = spline.rationalize(y_hom) # in physical coordinates

# Reference configuration:
X = spline.F

# Current configuration:
x = X + y

# Return a 3D elastic strain energy density, given E in Cartesian coordinates.
def psi_el(E):
    # Neo-Hookean potential, as an example:
    mu = Constant(1e4)
    C = 2.0*E + Identity(3)
    I1 = tr(C)
    return 0.5*mu*(I1 - 3.0)

# Shell thickness:
h_th = Constant(0.03)

# Obtain a through-thickness integration measure:
N_QUAD_PTS = 4
dxi2 = throughThicknessMeasure(N_QUAD_PTS,h_th)

# Potential energy density, including Lagrange multiplier term for
# incompressibility:
psi = incompressiblePotentialKL(spline,X,x,psi_el)

# Total internal energy:
Wint = psi*dxi2*spline.dx

# Take the Gateaux derivative of Wint in test function direction z_hom.
z_hom = TestFunction(spline.V)
z = spline.rationalize(z_hom)
dWint = derivative(Wint,y_hom,z_hom)

# External follower load magnitude:
PRESSURE = Constant(5e2)

# Divide loading into steps to improve nonlinear convergence.
N_STEPS = 100
DELTA_T = 1.0/float(N_STEPS)
stepper = LoadStepper(DELTA_T)

# Parameterize loading by a pseudo-time associated with the load stepper.
_,_,_,_,A,_ = surfaceGeometry(spline,X)
_,_,a2,_,a,_ = surfaceGeometry(spline,x)
dWext = -(PRESSURE*stepper.t)*sqrt(det(a)/det(A))*inner(a2,z)*spline.dx

# Full nonlinear residual:
res = dWint + dWext

# Consistent tangent:
Dres = derivative(res,y_hom)

# Allow many nonlinear iterations.
spline.maxIters = 100

# Files for output:  Because an explicit B-spline is used, we can simply use
# the homogeneous (= physical) representation of the displacement in a
# ParaView Warp by Vector filter.

d0File = File("results/disp-x.pvd")
d1File = File("results/disp-y.pvd")
d2File = File("results/disp-z.pvd")

# Use SNES solver for line search, to contend with extreme nonlinearity:
problem = ExtractedNonlinearProblem(spline,res,Dres,y_hom)
solver = PETScSNESSolver()
solver.parameters["linear_solver"] = "mumps"
solver.parameters["line_search"] = "bt"
solver.parameters["relative_tolerance"] = 1e-3
extSolver = ExtractedNonlinearSolver(problem,solver)

# Iterate over load steps.
for i in range(0,N_STEPS):
    if(mpirank==0):
        print("------- Step: "+str(i+1)+" , t = "+str(stepper.tval)+" -------")

    # Execute nonlinear solve.
    #spline.solveNonlinearVariationalProblem(res,dRes,y_hom)
    extSolver.solve()

    # Advance to next load step.
    stepper.advance()

    # Output solution.
    (d0,d1,d2) = y_hom.split()
    d0.rename("d0","d0")
    d1.rename("d1","d1")
    d2.rename("d2","d2")
    d0File << d0
    d1File << d1
    d2File << d2
    
####### Postprocessing #######

# Because we are using an explicit B-spline with unit weights on all control
# points and equal physical and parametric spaces, it is sufficient to simply
# load all three displacement files, apply the Append Attributes filter to
# combine them all, use the Calculator filter to produce the vector field
#
#  d0*iHat + d1*jHat + d2*kHat
#
# and then apply the Warp by Vector filter.
