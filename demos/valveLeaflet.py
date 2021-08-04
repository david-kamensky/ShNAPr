"""
This demo illustrates the usage of the ``ContactPlane`` class from 
``ShNAPr.contact``, by simulating quasi-static pressurization of a simplified
heart valve leaflet.  An archive with data files defining the valve geometry 
may be downloaded from the following link:

  https://www.dropbox.com/s/ot5i568dw40h75c/leaflet-geometries.tgz?dl=1

Extract this archive into the working directory in which you want to run this
demo, using the command ``tar cvzf leaflet-geometries.tgz``.  This contains
geometry for three leaflets, because it is shared by a separate demo 
distributed with the CouDALFISh library.  However, only the file 
``smesh.1.dat`` is used here, for a single-leaflet analysis.
"""

from tIGAr.BSplines import *
from tIGAr.timeIntegration import *
from ShNAPr.kinematics import *
from ShNAPr.SVK import *
from ShNAPr.contact import *
from os import path
import math

# Check whether the user downloaded and extracted the necessary data file to
# the working directory.
fnamePrefix = "smesh."
fnameSuffix = ".dat"
fnameCheck = fnamePrefix+"1"+fnameSuffix
import os.path
if(not os.path.isfile(fnameCheck)):
    if(mpirank==0):
        print("ERROR: Missing data file for valve geometry. "
              +"Please refer to the docstring at the top of this script.")
    exit()

####### Preprocessing #######

if(mpirank==0):
    print("Generating extraction data...")
    
# Load a control mesh given in legacy ASCII format.
controlMesh = LegacyMultipatchControlMesh(fnamePrefix,1,fnameSuffix)

# Create an equal-order spline with three displacement components as
# unknown fields.
splineGenerator = EqualOrderSpline(3,controlMesh)

# Apply pinned boundary conditions on the attached edge.
N_LAYERS = 1 # (Set to 2 for clamped boundary condition.)
scalarSpline = splineGenerator.getControlMesh().getScalarSpline()
patch = 0
for side in range(0,2):
    for direction in range(0,2):
        if(not (direction==1 and side==0)):
            sideDofs = scalarSpline\
                       .getPatchSideDofs(patch,direction,side,
                                         nLayers=N_LAYERS)
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

# The shell thickness:
h_th = Constant(0.03)

# The Young's modulus and Poisson ratio:
E = Constant(1e7)
nu = Constant(0.45)

# Elastic energy:
Wint = surfaceEnergyDensitySVK(spline,X,x,E,nu,h_th)*spline.dx

# Take the Gateaux derivative of Wint in test function direction z_hom.
z_hom = TestFunction(spline.V)
z = spline.rationalize(z_hom)
dWint = derivative(Wint,y_hom,z_hom)

# Maximum magnitude of external follower load:
PRESSURE = Constant(1e5)

# Divide loading into steps to improve nonlinear convergence.
N_STEPS = 10
DELTA_T = 1.0/float(N_STEPS)
stepper = LoadStepper(DELTA_T)

# Parameterize loading by a pseudo-time associated with the load stepper.
_,_,_,_,A,_ = surfaceGeometry(spline,X)
_,_,a2,_,a,_ = surfaceGeometry(spline,x)
areaJac = sqrt(det(a)/det(A))
dWext = -(PRESSURE*stepper.t)*areaJac*inner(a2,z)*spline.dx

# Define two contact planes to model effects of other leaflets:
origin = Constant((0,0,0))
s32 = 0.5*math.sqrt(3.0)
n0 = Constant((-s32,-0.5,0))
n1 = Constant((-s32, 0.5,0))
k = Constant(1e9)
smoothingDist = Constant(0.015)
plane0 = ContactPlane(origin,n0,k)
plane1 = ContactPlane(origin,n1,k)

# Define virtual work contributions of contact planes:
f0 = plane0.forceDensity(x)
f1 = plane1.forceDensity(x)
dWcont0 = -dot(f0,z)*areaJac*spline.dx
dWcont1 = -dot(f1,z)*areaJac*spline.dx
dWcont = dWcont0 + dWcont1

# Full nonlinear residual:
res = dWint + dWext + dWcont

# Consistent tangent:
Dres = derivative(res,y_hom)

# Files for output:  

# For x, y, and z components of displacement:
d0File = File("results/disp-x.pvd")
d1File = File("results/disp-y.pvd")
d2File = File("results/disp-z.pvd")

# For x, y, and z components of initial configuration:
F0File = File(selfcomm,"results/F-x.pvd")
F1File = File(selfcomm,"results/F-y.pvd")
F2File = File(selfcomm,"results/F-z.pvd")

# For weights:
F3File = File(selfcomm,"results/F-w.pvd")

# Use SNES solver for line search, to contend with extreme nonlinearity:
problem = ExtractedNonlinearProblem(spline,res,Dres,y_hom)
solver = PETScSNESSolver()
solver.parameters["linear_solver"] = "mumps"
solver.parameters["line_search"] = "bt"
solver.parameters["relative_tolerance"] = 1e-3
extSolver = ExtractedNonlinearSolver(problem,solver)

# Write all the output files for whatever displacement is currently in
# the Function y_hom.
def writeOutput():
    (d0,d1,d2) = y_hom.split()
    d0.rename("d0","d0")
    d1.rename("d1","d1")
    d2.rename("d2","d2")
    d0File << d0
    d1File << d1
    d2File << d2
    spline.cpFuncs[0].rename("F0","F0")
    spline.cpFuncs[1].rename("F1","F1")
    spline.cpFuncs[2].rename("F2","F2")
    spline.cpFuncs[3].rename("F3","F3")
    F0File << spline.cpFuncs[0]
    F1File << spline.cpFuncs[1]
    F2File << spline.cpFuncs[2]
    F3File << spline.cpFuncs[3]    

# Write the initial conditions:
writeOutput()
    
# Iterate over load steps.
for i in range(0,N_STEPS):
    if(mpirank==0):
        print("------- Step: "+str(i+1)+" , t = "+str(stepper.tval)+" -------")

    # Execute nonlinear solve.
    extSolver.solve()

    # Advance to next load step.
    stepper.advance()

    # Write output files for updated solution:
    writeOutput()
    
####### Postprocessing #######

# Notes for plotting the results with ParaView:
#
# Load the time series from all seven files and combine them with the
# Append Attributes filter.  Then use the Calculator filter to define the
# vector field
#
# ((d0+F0)/F3-coordsX)*iHat+((d1+F1)/F3-coordsY)*jHat+((d2+F2)/F3-coordsZ)*kHat
#
# which can then be used in the Warp by Vector filter.  To visualize the full
# valve, apply the Transform filter this, to create two copies, rotated by
# 120 and 240 degrees.  
