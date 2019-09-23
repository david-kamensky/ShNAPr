"""
This demo solves the Scordelis--Lo roof problem.  For simplicity, the full
roof is discretized, and the rigid mode that would normally be eliminated by
a symmetry boundary condition is eliminated by fixing the corresponding 
displacement component of one node.  To simplify geometry construction with
igakit, the y and z directions are also switched from their canonical order.
We take the precise problem setup and converged reference value for the 
quantity of interest from Section 6.2.1 of

  https://mediatum.ub.tum.de/doc/1002634/464162.pdf
"""

# NURBS-based IGA:
from tIGAr import *
from tIGAr.NURBS import *

# Shell problem definitions:
from ShNAPr.kinematics import *
from ShNAPr.SVK import *

# Geometry creation:
from igakit.cad import *
import math
from numpy import linspace, array

####### Parameters #######

# Discretization parameters come from command line:

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Nel',dest='Nel',default=32,
                    help="Number of elements along one side of domain.")
parser.add_argument('--p',dest='p',default=3,
                    help="Polynomial degree of basis; must be > 1.")
args = parser.parse_args()
p = int(args.p)
Nel = int(args.Nel)

# Fixed parameters for problem:
L = 50.0
R = 25.0
theta = 40.0 # (in degrees)
h_th = Constant(0.25)
E = Constant(4.32e8)
nu = Constant(0.0)
arealForceDensity = Constant(90.0)

####### Geometry creation using igakit #######

# Note: It is more convenient given the functions from igakit for "down" to
# be the negative y direction instead of the negative z direction.

if(mpirank==0):
    print("Creating geometry with igakit...")

def degToRad(theta):
    return theta*math.pi/180.0
angle = (degToRad(90.0-theta),degToRad(90.0+theta))
C = circle(center=[0,0,0], radius=R, angle=angle)
T = circle(center=[0,0,L], radius=R, angle=angle)
S = ruled(C,T)
S.elevate(0,p-2)
S.elevate(1,p-1)
newKnots = linspace(0,1,Nel+1)[1:-1]
S.refine(0,newKnots)
S.refine(1,newKnots)

####### Spline setup #######

if(mpirank==0):
    print("Creating spline...")

# Create an extraction generator for an equal-order spline with three fields
# for the displacement components:
splineMesh = NURBSControlMesh(S,useRect=False)
splineGenerator = EqualOrderSpline(3,splineMesh)

# Pin x and y at both ends of the cylinder:
for field in range(0,2):
    scalarSpline = splineGenerator.getScalarSpline(field)
    parametricDirection = 1
    for side in [0,1]:
        sideDofs = scalarSpline.getSideDofs(parametricDirection,side)
        splineGenerator.addZeroDofs(field,sideDofs)

# Pin z displacement of one control point to eliminate rigid mode:
field = 2
splineGenerator.addZeroDofs(field,[0,])
        
####### Analysis #######

if(mpirank==0):
    print("Performing analysis...")

spline = ExtractedSpline(splineGenerator,2*p)

# Kinematics:
y_hom = Function(spline.V)
X = spline.F
x = X + spline.rationalize(y_hom)

# Internal energy:
Wint = surfaceEnergyDensitySVK(spline,X,x,E,nu,h_th)*spline.dx

# Test function:
z_hom = TestFunction(spline.V)
z = spline.rationalize(z_hom)

# Residual:
dWint = derivative(Wint,y_hom,z_hom)
f = as_vector([Constant(0.0),-arealForceDensity,Constant(0.0)])
res = dWint - inner(f,z)*spline.dx
Dres = derivative(res,y_hom)

# Geometrically-linear solution (i.e., one Newton iteration,
# starting from zero):
spline.solveLinearVariationalProblem(Dres==-res,y_hom)

####### Output #######

# Check the quantity of interest; its reference value comes from Kiendl's
# thesis, linked above.
xi = array([0.0,0.5])
QoI = -y_hom(xi)[1]/spline.cpFuncs[3](xi)
print("Quantity of interest = "+str(QoI)+" (Reference value = 0.3006)")

# Output of control mesh:
for i in range(0,3+1):
    name = "F"+str(i)
    spline.cpFuncs[i].rename(name,name)
    File("results/"+name+"-file.pvd") << spline.cpFuncs[i]

# Output of homogeneous displacement components:
y_hom_split = y_hom.split()
for i in range(0,3):
    name = "y"+str(i)
    y_hom_split[i].rename(name,name)
    File("results/"+name+"-file.pvd") << y_hom_split[i]

# Notes on visualization with ParaView:  First, load all seven output files,
# then apply an AppendAttributes filter to combine them.  Next use a
# Calculator filter with the formula
#
#  (F0/F3-coordsX)*iHat + (F1/F3-coordsY)*jHat + (F2/F3-coordsZ)*kHat
#
# to get the un-deformed configuration, and apply a WarpByVector filter, with
# scale 1.  After that, use another Calculator filter to get the displacement,
#
#  (y0/F3)*iHat + (y1/F3)*jHat + (y2/F3)*kHat
#
# which may be used in a second WarpByVector filter.  The second warp filter
# may then be scaled to taste.
