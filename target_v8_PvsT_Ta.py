#--------------------------------------------------------------------------------------
#	SOLVE 3D HEAT TRANSFER PROBLEM WITH EVAPORATION (VAPOR PRESSURE)
#--------------------------------------------------------------------------------------
from __future__ import print_function
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from dolfin import *
import ufl
from ufl import as_tensor
from ufl import Index
import math
from scipy.optimize import curve_fit
from mpi4py import MPI
from scipy.interpolate import interp1d
from scipy import interp
from scipy.interpolate import UnivariateSpline

parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
set_log_level(50)

file_total = 'Results/Ta_PvsT_3mm_alleffects_R=0.87.txt'


with open(file_total, 'w') as d:
	d.write('Simulation of Ta, with evaporation, kinetic energy and radiation, alpha = 1.16e+8, epsilon = 0.5*e(T)' + '\n')
	d.write('Spot Radius = 750um' + '\n')
	d.write('Reflectivity = 0.87' + '\n')
	d.write('Disk or Fiber: Disk (1030nm)' + '\n')
	d.write('T-dependent constants?: Yes' + '\n')
	d.write('If yes, which?: All ' + '\n')
	d.write('MP (K): 3293' + '\n')
	d.write('Diameter of Source: 3mm' + '\n')
	d.write('Length of Source: 8mm' + '\n')
	d.write('End Time: 200' + '\n')
	d.write('# of time steps: 400' + '\n')
	d.write('-----------------------------------' + '\n')
	d.write('Power(W)'+"\t"+'Average Source Temperature (K)' + '\t' + 'Evaporated Energy (enthalpy) (W)' + '\t' + 'Kinetic Energy to evaporant (W)'+ '\t' +'Transmission' + '\t' + 'Rad Power (W)' + '\t' + 'Evaporation Flux (mg/s)' + '\n')
#----------------
t_start = 0.0
t_end = 200
nstep = 400
dtime = (t_end - t_start)/ nstep  #time step

#---------------------------------------

T_am = 300 #ambient vacuum temperature (K)
T_a4 = T_am**4
sigma = 5.67E-8 # W/(m**2.K**4) S-B Constant
w = 750e-6  #m  width of guassian laser (750um for 60mm working distance)
deltaH = 753e3 #J/mol
M_molar = 180.95e-3 #kg/mol
ideal_gas = 8.31446261815324
k_b = 1.380649e-23 # J/K
pi = 3.141592653589793
mass_Ta = 180.95 * (1.66e-27) # kg
N_A = 6.02214E23 #/mol avagardros number
T_crit = 13400 #K

deltaH_m = deltaH/M_molar #J/kg

#------Import Constants data------

kappa_data =  np.genfromtxt("Constants/Ta-kappa.txt", skip_header=2, delimiter="	", dtype = float)
c_data = np.genfromtxt("Constants/Ta-cp.txt", skip_header=2, delimiter="	", dtype = float)
rho_data = np.genfromtxt("Constants/Ta-density.txt", skip_header=3, delimiter="	", dtype = float)
emiss_data = np.genfromtxt("Constants/Ta-emissivity.txt", skip_header=2, delimiter="	", dtype = float)

kappa_in = kappa_data[:,0]
temp = kappa_data[:,1]
tempspace = np.linspace(200,10000,10000)
kappa_f = UnivariateSpline(temp, kappa_in, k = 1, ext = 3)

c_p_in = c_data[:,0]
temp_c = c_data[:,1]
c_f = UnivariateSpline(temp_c, c_p_in, k = 2, ext = 3)

rho_in = rho_data[:,0]
temp_rho = rho_data[:,1]
rho_f = UnivariateSpline(temp_rho, rho_in, k = 1, ext = 3)

emiss_in = emiss_data[:,0] * 0.5 
temp_emiss = emiss_data[:,1]
emiss_f = UnivariateSpline(temp_emiss, emiss_in, k = 2, ext = 3)



#---------IMPORT EVAPORATION DATA--------------
def vp_function(x,a):
	return a*np.exp(-(deltaH/ideal_gas)*1/(x))

file_vp = 'Constants/Ta-VP.txt'
vp_data = np.genfromtxt(file_vp, skip_header=1, delimiter="	", dtype = float)
vp_in = vp_data[:,2]
temp_vp = vp_data[:,0]
popt_vp, pcov_vp = curve_fit(vp_function, temp_vp, vp_in,p0 = [1e9])
vp_f = vp_function(tempspace, *popt_vp)
#vp_f = UnivariateSpline(temp_vp, vp_in, k = 2, ext = 3)

fig=plt.figure(figsize=(4.5,3.6))
ax=fig.add_subplot(1,1,1)
ax.minorticks_on() # enable minor ticks
ax.set_axisbelow(True) # put grid behind curves
ax.grid(b=True, which='major', color='black', linestyle='-', zorder=1, linewidth=0.4, alpha = 0.12) # turn on major grid
ax.grid(b=True, which='minor', color='black', linestyle='-', zorder=1, linewidth=0.4, alpha = 0.12) # turn on minor grid
ax.scatter(temp_vp,vp_in, color = 'black', label = 'data', s= 5, zorder = 3)
ax.plot(tempspace, vp_f, color = 'blue', label = 'Fit', zorder = 2)
ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_xlabel('Temperature [K]')
ax.set_ylabel('Vapor Pressure [Pa]')
ax.legend(labelspacing=0.25, fontsize = 8)
plt.xlim([1000,5000])
plt.ylim([1e-10,1e5])
plt.savefig('Results/Ta_vpdata.pdf', bbox_inches='tight', format='pdf')
plt.savefig('Results/Ta_vpdata.png', dpi=300, bbox_inches='tight', format='png')

#---------------------------------------
data = 'mesh_3mm'

mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), data+'.h5', 'r')
hdf.read(mesh, '/mesh', False)
cells_f = MeshFunction('size_t', mesh, 3)
hdf.read(cells_f, '/cells')
facets_f = MeshFunction('size_t', mesh, 2)
hdf.read(facets_f, '/facets')
boundaries = facets_f
#boundaries = MeshFunction('size_t', mesh, 'crucible_facet_region.xml') 
#-----------------------------------------------
# Mark the tags for 'top' and 'surface' groups
#-----------------------------------------------
tag_s = 2  #Tags for 'surface' group in .msh file
tag_t = 3
surfacefacetindices = np.where(boundaries.array() == tag_s)[0]
surfacefacetindices_2 = np.where(boundaries.array() == tag_t)[0]
surfacefacetindices_total = np.append(surfacefacetindices,surfacefacetindices_2)

surfacefacets = np.array(list(facets(mesh)))[surfacefacetindices_total]

coordinates = []
normalvec = []
    
for f in surfacefacets:
	coordinates.append(f.midpoint().array())
	normalvec.append(f.normal().array())
#print(coordinates[0])

all_cells = np.array(list(cells(mesh)))

tdim = mesh.topology().dim()
fdim = tdim - 1
mesh.init(fdim, tdim)
f_to_c = mesh.topology()(fdim, tdim)
c_to_f = mesh.topology()(tdim, fdim)


area_fac = []
for facet in surfacefacets:
	cell = all_cells[f_to_c(facet.index())[0]]
	local_facets = c_to_f(cell.index())
	local_index = np.flatnonzero(local_facets == facet.index())
	area_fac.append(cell.facet_area(local_index))


area = pi * w**2
#------------
Pmin = 60
Pmax = 2000
npstep = 30

for p in np.logspace(np.log10(Pmin), np.log10(Pmax), npstep):
	P_in = p
	I = P_in / (area)    #Intensity 


	i,j,k = ufl.indices(3)
	n = FacetNormal(mesh)


	Reflect = 0.87
	absorb = 1 - Reflect
	x = SpatialCoordinate(mesh)

	Laser = 1*(absorb)*I*exp(-pow((x[0] - 0), 2)/(w*w)-pow((x[1]-0), 2)/(w*w))

# function based off images (side view) to account for intensity

#------------------------------------------------

	VectorSpace = VectorFunctionSpace(mesh, 'P', 1)
	da = Measure('ds', domain=mesh, subdomain_data = facets_f, metadata = {'quadrature_degree': 2})  #area element

	dv = Measure('dx', domain=mesh, subdomain_data = cells_f, metadata = {'quadrature_degree': 2})   #volume element

	Space = FunctionSpace(mesh, 'P', 1) 
	T = Function(Space)
	T0 = Function(Space)
	T_init = Expression('Tambient', degree=1, Tambient=300.)
	T = project(T_init, Space)
	assign(T0,T)

	
#------------------------------------------------
# We should set up function spaces for each constant that will change as a function of temperature
#------------------------------------------------

	k_new_array = kappa_f(T.vector().get_local())
	kappa_space = FunctionSpace(mesh, 'CG',1)
	kappa = Function(kappa_space)
	kappa.vector()[:] = k_new_array

	rho_new_array = rho_f(T.vector().get_local())
	rho_space = FunctionSpace(mesh, 'CG',1)
	rho = Function(rho_space)
	rho.vector()[:] = rho_new_array

	c_new_array = c_f(T.vector().get_local())
	c_space = FunctionSpace(mesh, 'CG',1)
	c = Function(c_space)
	c.vector()[:] = c_new_array

	emiss_eff_array = emiss_f(T.vector().get_local())
	emiss_space_eff = FunctionSpace(mesh, 'CG',1)
	epsilon_eff = Function(emiss_space_eff)
	epsilon_eff.vector()[:] = emiss_eff_array

#--------------FOR REFLECT = 1-EMISS---------------
	#absorb_array = emiss_f(T.vector().get_local())
	#absorb_space_eff = FunctionSpace(mesh, 'CG',1)
	#absorb = Function(absorb_space_eff)
	#absorb.vector()[:] = absorb_array
#----------------------------------------------------

	def Evap3(T): #kg/m^2-sec
		return popt_vp[0]*exp(-(deltaH/ideal_gas)*1/(T))*(mass_Ta/(2*pi*k_b*T))**0.5
	evap_rate = Evap3(T)

	def attenuation(T):
		pressure = popt_vp[0]*exp(-(deltaH/ideal_gas)*1/(T))
		#return exp(-(1.2021e+8*0.098*pressure)/(np.sqrt(2)*2700*ideal_gas*T))
		return exp(-(1.16e+8*0.098*M_molar*pressure)/(np.sqrt(2)*16690*ideal_gas*T))
	atten = attenuation(T)
#-------------------------------------------

	V = TestFunction(Space)     # Test Function used for FEA
	dT = TrialFunction(Space)   # Temperature Derivative
	q0 = Function(VectorSpace)  # heat flux at previous time step
	i = Index()
	G = as_tensor(T.dx(i), (i))  #gradient of T
	G0 = as_tensor(T0.dx(i), (i)) # gradient of T at previous time step 
	
	q = -kappa*G
	
	F = (rho*c/dtime*(T-T0)*V - q[i]*V.dx(i)) * dv + epsilon_eff*sigma*(T**4 - T_a4)*V*da + evap_rate*deltaH_m*V*da + (3.0/2.0)*T*evap_rate*((N_A*k_b)/M_molar)*V*da - atten*Laser*V*da(tag_t)  #final form to solve (free standing source)
	
	Gain = derivative(F, T, dT)    # Gain will be used as the Jacobian required to determine the evolution of a dynamic system 


	problem = NonlinearVariationalProblem(F, T, [], Gain)
	solver  = NonlinearVariationalSolver(problem)

	solver.parameters["newton_solver"]["relative_tolerance"] = 1E-4
	solver.parameters["newton_solver"]["absolute_tolerance"] = 1E-3
	solver.parameters["newton_solver"]["convergence_criterion"] = "residual"
	solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
	solver.parameters["newton_solver"]["linear_solver"] = "cg"
	solver.parameters["newton_solver"]["maximum_iterations"] = 10000
	solver.parameters["newton_solver"]["preconditioner"] = "hypre_euclid"
	solver.parameters["newton_solver"]["report"] = True

	solver.parameters["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = False
	solver.parameters["newton_solver"]["krylov_solver"]["relative_tolerance"] = 1E-4
	solver.parameters["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1E-3
	solver.parameters["newton_solver"]["krylov_solver"]["monitor_convergence"] = False
	solver.parameters["newton_solver"]["krylov_solver"]["maximum_iterations"] = 100000

	#file_T = File('/home/tsmart/Data/Fenics/target-v8-SourceGeometry/Results/2mm/solution.pvd')

	t = 0
	mass_total = 0 #total evaporated mass
	#file_T << (T,t)
	is_looping = True
	for t in np.arange(t_start + dtime,t_end + dtime,dtime):
		if (mass_total < 5e-4):
			print( "Time", t)
			solver.solve()
		#--------------------------------------------
			k_new_array = kappa_f(T.vector().get_local())
			kappa.vector()[:] = k_new_array

			rho_new_array = rho_f(T.vector().get_local())
			rho.vector()[:] = rho_new_array

			#c_new_array = c_f(T.vector().get_local())
			#c.vector()[:] = c_new_array

			emiss_eff_array = emiss_f(T.vector().get_local())
			epsilon_eff.vector()[:] = emiss_eff_array
		#--------------------------------------------
			transmission = assemble(atten*da(tag_t))/assemble(1*da(tag_t))
			#print(transmission)
			max_T = max(T.vector())
			print("Maximum temperature is: " + str(max_T) + 'K')
			epsilon_max = max(epsilon_eff.vector())
			print("Epsilon max is: "+ str(epsilon_max))
			#file_T << (T,t)
			assign(T0, T)
			#print(max(T.vector()), 'K')
			#print('Evaporated Energy: ', assemble(evap_rate*deltaH_m*da(3)), 'W')
		else:
			is_looping = False
			break
		if not is_looping:
			break
	average_T = assemble(T*da(tag_t))/assemble(1*da(tag_t))
	max_T = max(T.vector())

	MPI.COMM_WORLD.barrier()


	evapenergy = assemble(evap_rate*deltaH_m*da(tag_t))
	kineticenergy = assemble((3.0/2.0)*T*evap_rate*((N_A*k_b)/M_molar)*da(tag_t))
	rad_power = assemble(epsilon_eff*sigma*(T**4 - T_a4)*da) # power radiated off surfaces
	evaprate = assemble(1e6*evap_rate*da(tag_t))
#print('Top Rad Power:', rad_power)
#print('Total Rad Power:', rad_power_total)


	if MPI.COMM_WORLD.rank ==0:
		with open(file_total, "a") as d:
			d.write(str(P_in) + "\t" + str(max_T) + '\t' + str(evapenergy) + '\t' + str(kineticenergy) + '\t' + str(transmission) + '\t' + str(rad_power) + '\t' + str(evaprate) + '\n')
