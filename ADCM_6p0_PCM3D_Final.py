# Author: Diego S. Prado

import os as oslib
import time as tm
from dolfin import *
import numpy as np
from mshr import *
from mpi4py import MPI
import math as mt
from shutil import copy2


from scipy.interpolate import spline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpltools import style

comm = MPI.COMM_WORLD   
rank = comm.Get_rank()
set_log_level(50)
#%%
##### MEF Parameters #####
n = 60

theta = 1.0                 # Time-derivative [0: Forward; 1: Backward; 0.5: Crank]
smootime = 0.1              # length of transition time of inlet pressure, in relation to timef
mtFact = Constant(1E-7)             # Melting time factor
penPcm = Constant(1.0)

p = 1.0                     # penalization factor
q_degree = 6                # quadrature points

solver_type = "krylov"        # solver
solver_method = "gmres"       # method
solver_precond = "hypre_amg"  # pre-conditioner
solver_reltol = 1e-5          # relative tolerance
solver_maxtol = 1e-5

##### Times #####
allTimes = np.concatenate((np.linspace(0.0, 30, 90),np.linspace(31.0, 300, 300)), axis=0)

tTol = 0.25                 # K
gamma = 75.0

minTimeStep = 0.0001
maxTimeStep = 1.0

factorTime = 0.75

time_disc_method = 'bdf1' # 1 - BDF1 | 2 - BDF2

##### TOM Parameters #####
Pcmmin = Constant(0)                # minimum porosity volume allowed
Pcmmax = Constant(1)                # maximum porosity volume allowed
opit = 25                           # iterations in optimizer

####### Chart Config #######
tIni = 233.0
tStep = 0.01

##### Results #####
hAt = 0 # Export files with measures at specific node (0: Don't Export)
keyres = 10.0 # Save files every [keyres] seconds

#%%
##### Properties #####
# Gas - Methane
gas_cp = Constant(2450.0)       # specific heat at 300K [J/kg.K]
gas_k = Constant(0.0343)      # thermal conductivity [W/m.K] 
gas_mg = Constant(0.016)       # molar mass [kg/mol]
gas_pcr = Constant(4.596e6)     # critical pressure [Pa]
gas_tcr = Constant(120.0)      # critical temperature [K]
gas_mu = Constant(1.25e-5)      # viscosity [Pa.s]
gas_rhoads = Constant(422.62)    # adsorbed density [kg/m^3]

# Adsorbent - Norit
ads_cp = Constant(650.0)        # Adsorbent specific heat [J/kg.K]
ads_k = Constant(0.54)          # Adsorbent thermal conductivity [W/m.K]
ads_ws = Constant(3.3e-4)       # Adsorbent specific micropore volume [m^3/kg]
ads_ns = Constant(1.6)          # Adsorbent micropore dispersion
ads_rhos0 = Constant(1428.57)     # Adsorbent density [kg/m^3]
#ads_rhos0 = Constant(944.50)     # Xingcun Li
ads_dps = Constant(28.3e-6)     # Adsorbent particle diameter [m]
ads_K = Constant(3.7e-9)       # Adsorbent Permeability
ads_emi = Constant(0.7)    # Adsorbent Microporosity
#ads_emi = Constant(0.72)    # Adsorbent Microporosity
ads_ema = Constant(0.0)    # Adsorbent Macroporosity

# Methane - Norit
E0 = Constant(25040.60)     # characteristic energy of adsorption [J/mol]
Ea = Constant(6000.0)       # energy of adsorption [J/mol]
Ed = Constant(22000.0)      # energy of desorption [J/mol]
beta = Constant(0.35)       # affinity coefficient
DelH = Constant(-13300.0)   # reaction enthalpy [J/mol]
DelS = Constant(-87.84)     # reaction entropy [J/mol.K]
D = Constant(3.2)           # kinetic constant of adsorption [1/s]      
ModDelH = -DelH             # modulus of reaction enthalpy [J/mol] 
Rg = Constant(8.31)         # universal gas constant [J/mol.K]
Tb = Constant(111.2)
alphae = Constant(2.5e-3)
A1 = Constant(4000.0)
B1 = Constant(1.57)
C1 = Constant(2.4e-8)
D1 = Constant(749.0)
posx = Constant(340000)
posy = Constant(-10.2)


# PCM - HS34
tpc = Constant(313.0)      # phase change temperature [K]
rhopcm = Constant(909.45)   # density [kg/m^3]
Lat = Constant(153000.0)    # latent heat [J/kg]
kpcms = Constant(0.460)       # thermal conductivity [W/m.K]
kpcml = Constant(0.460)       # thermal conductivity [W/m.K]
Cpcms = Constant(2200.0)      # specific heat - solid [J/kg.K]
Cpcml = Constant(2200.0)      # specific heat - liquid [J/kg.K]
epsPC = Constant(6.5)       # material model coeficient
poropcm = Constant(0.0)     # pcm porosity

# Copper
kcopp = Constant(401.0)      # thermal conductivity [W/m.K]
Ccopp = Constant(380.0)      # specific heat - solid [J/kg.K]
rhocopp = Constant(8.954e3)  # density [kg/m^3]

# Room
Rg = Constant(8.31)
troom = Constant(300.0)       # room temperature [K]
patm = Constant(100000.0)     # atm pressure [Pa]
hroom = Constant(5.0)    # convective coefficient [W/m^2.K] 5 or 700
htube = Constant(400.0)     # convective coefficient for inner tubes [W/m^2.K]
hentrada = Constant(1000.0)
pfator = Constant(1000.0)

#%%  
# Initial Conditions
# Adsorption
Pini = Constant(100000)     # Initial State Pressure
Tini = Constant(300.0)
Rini = Constant(0.0)

#%%
# Temperature Measure Points
tMP1 = Point(0.008,0.0,0.003)
tMP2 = Point(0.0935,0.0,0.003)
tMP3 = Point(0.179,0.0,0.003)
tMP4 = Point(0.2645,0.0,0.003)
tMP5 = Point(0.350,0.0,0.003)

#%%
# Material Distribution

#class Pcmmap(Expression):
#    "Experimental - Porosity Map"
#    def eval(self, value, x):
#        material = 0
#        if (x[0] > 0.07 and x[0] < 0.22) and (x[1] < 0.0275 and x[1] > 0.025):
#            material = 1
##            elif x[1] >= 0.018 and x[1] <= 0.023:
##                material = 1
##            elif x[1] >= -0.023 and x[1] <= -0.018:
##                material = 1
#        value[0] = 1.0 if material == 1 else 0.0
      
      
class Pcmmap(Expression):
    "Experimental - Porosity Map"
    def eval(self, value, x):
        material = 0
        if (x[0] > 0.02451 and x[0] < 0.3745001) and (x[1]**2 + x[2]**2 < 0.0024**2 or x[1]**2 + (x[2]-0.02)**2 < 0.0024**2):
            material = 1
#            elif x[1] >= 0.018 and x[1] <= 0.023:
#                material = 1
#            elif x[1] >= -0.023 and x[1] <= -0.018:
#                material = 1
        value[0] = 1.0 if material == 1 else 0.0

class copperMap(Expression):
    "Experimental - Porosity Map"
    def eval(self, value, x):
        material = 0
        if (x[0] > 0.02501 and x[0] < 0.375001) and (x[1]**2 + x[2]**2 < 0.0031**2 or x[1]**2 + (x[2]-0.02)**2 < 0.0031**2):
            material = 1
#            elif x[1] >= 0.018 and x[1] <= 0.023:
#                material = 1
#            elif x[1] >= -0.023 and x[1] <= -0.018:
#                material = 1
        value[0] = 1.0 if material == 1 else 0.0        

#%% Functions
# Epsilon t
def epsTotal(epsmet,epspcm):
    "Total porosity"
    return (ads_ema + (1 - ads_ema)*ads_emi)*(1-epsmet) *(1-epspcm)

# Ideal Gas Law    

def adsVarCp(t):
    return (Constant(-23.37) + Constant(0.25696011)*t - Constant(0.000883376)*(t**2) + Constant(1.02349e-6)*(t**3))*Constant(1000.0)
    
def gas_rho(p,t):
    return (gas_mg*p)/(Rg*t)
    
def Keff(epsmet,epspcm):
    return ads_K*(1-epsmet)*(1-epspcm) + Constant(1.0e-17)*epsmet + Constant(1.0e-17)*epspcm

def rhoads_T(t):
    return gas_rhoads/(exp(alphae*(t - Tb)))

def Psat(t):
    return gas_pcr*(t/gas_tcr)**2
    
def A2(p,t):
    return Rg*t*ln(Psat(t)/(p))
    
def Qeq(p,t,epsmet,epspcm):
#    return (((C1*exp(D1/t))*(p+posx)*(A1*t**(-B1)))/(1+(C1*exp(D1/t))*(p+posx))-(posy/rhos0))*(1-eps_pcm)*(1-epsc)
    return (rhoads_T(t) * Wf() * exp(-(A2(p,t)/(beta*E0))**ads_ns))*(1-epsmet)*(1-epspcm)*Constant(2.0)
#    return (((C1*exp(D1/t))*p*(A1*t**(-B1)))/(1+(C1*exp(D1/t))*p))*rhoads_T(t)*Wf()*(1-epsmet)*(1-epspcm)
    
def Wf():
    return ads_ws
    
def Gvar(t):
    return D * exp(-Ea/(Rg*t))
#    return Constant(0.000032)
    
def Ceff(q,p,t,epsmet,epspcm):
    return ((((epsTotal(epsmet,epspcm)*gas_mg*p/(Rg*t))+(Constant(1.0-epsTotal(epsmet,epspcm)))*ads_rhos0*q)*gas_cp + (Constant(1.0-epsTotal(epsmet,epspcm))*ads_rhos0*adsVarCp(t)))*(1-epsmet)*(1-epspcm))+Ccopp*rhocopp*(epsmet)+Cpcm(t)*rhopcm*(epspcm)

def keff(epsmet,epspcm,t):
    return ((gas_k*epsTotal(epsmet,epspcm)+(1-epsTotal(epsmet,epspcm))*ads_k)*(1-epsmet)*(1-epspcm))+kcopp*(epsmet)+kpcm(t)*(epspcm)
    
#Linearized Beta Function -> Polynomial
def linBeta(T):
    return  Constant(-5862.72267936096) + Constant(80.95115153507103)*T + Constant(-0.4468225014325379)* T**2 + Constant(0.0012322506626282188)* T**3 + Constant(-1.6976774955807072*(10**-6))* T**4 + Constant(9.346725300406145*(10**-10))* T**5           

# If - for temperature of phase change
def meltT(T):
    return Constant(0.5) + tanh(Constant(0.1)*(T**2 - tpc**2))/Constant(2.0)
    
# If - for temperature of phase change
def meltLim(hl,hlim):
    return Constant(0.5) - tanh(Constant(0.1)*(hl**2 - hlim**2))/Constant(2.0)

#Heat Flux for Latent            
def heatFluxLat(T,mtime,hl,hlim,epspcm):
    return -rhopcm * Lat * linBeta(T) * meltT(T) * sqrt(kpcml/(Cpcml*rhopcm))/sqrt(sqrt(mtime**2)+mtFact) * meltLim(hl,hlim) *epspcm

#Storing Heat in Phase Change
def phaseChangeStore(T,ql0,hl,hlim,dtime,mtime,epspcm):
    return -heatFluxLat(T,mtime,hl,hlim,epspcm)*(dtime)*epspcm
    
#Melting Time
def meltingTime(ql,T,cArea,epspcm):
    return -((ql/(mtFact + Constant(2.0) * rhopcm * Lat * epspcm * linBeta(T) * meltT(T) * sqrt(kpcms/(Cpcms*rhopcm))))**2) * meltT(T -  Constant(0.2))*penPcm


#%% Phase Change Material Model
# Phase Change Functions 
def Delphi(t):
    "Phase Changing Volume"
    return (e**(-(epsPC**2)*(t-tpc)**2)*epsPC*pi**(-1/2))
    
def Phi(t):
    "Phase Changed Volume"
    return (0.5+0.5*erf(epsPC*(t-tpc)))

# Thermal Properties Functions
def Cpcm(t):
    "Apparent Specific Heat"
    return Cpcms + (Cpcml - Cpcms)*Phi(t) + Lat * Delphi(t)
# Effective k    
def kpcm(t):
    'Thermal Conductivity'
    return kpcms + (kpcml - kpcms)*Phi(t)

# Efective Rho   
#def rhoe(eps):
#    'Thermal Conductivity'
#    return rho*eps+rhoc*(1-eps)


#%% Plotting 
def plotRes(xdata,ydata,details,xlabel,ylabel,savefolder,filename):
	# Style
	style.use('ggplot')
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

#	if self.D.interpolate:
#		self.D.x_o = self.D.x
#		self.D.x = np.linspace(self.D.x_o.min(),self.D.x_o.max(),self.D.interpstep)

	fig = plt.figure()
	for i in range(len(ydata)):
#		if self.D.interpolate:
#			self.D.y[i] = spline(self.D.x_o, self.D.y[i], self.D.x)

		plt.plot(
			xdata, 
			ydata[i], 
			color = details[i]['color'], 
			linestyle = details[i]['linestyle'], 
			marker = details[i]['marker'], 
			markevery = 1, 
			markeredgecolor = 'none', 
			label = details[i]['label']
		)

#	if self.D.limited_axes:
#		plt.axis(self.D.axes_limits)

	# plt.fill_between(self.D.x, self.D.y[0], self.D.y[1], facecolor='yellow', alpha=0.5)

#		plt.axis(self.D.axes_limits)

	# plt.fill_between(self.D.x, self.D.y[0], self.D.y[1], facecolor='yellow', alpha=0.5)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	plt.legend(loc='best')

	fig.savefig(savefolder+filename, format='eps')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	plt.legend(loc='best')


class Chart:
	def __init__(self,D):
		self.D = D

	def plot(self):
		# Style
		style.use('ggplot')
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

#		if self.D.interpolate:
#			self.D.x_o = self.D.x
#			self.D.x = np.linspace(self.D.x_o.min(),self.D.x_o.max(),self.D.interpstep)

		self.fig = plt.figure()
		for i in range(len(self.D.y)):
#			if self.D.interpolate:
#				self.D.y[i] = spline(self.D.x_o, self.D.y[i], self.D.x)

			plt.plot(
				self.D.x, 
				self.D.y[i], 
				color = self.D.y_prop[i]['color'], 
				linestyle = self.D.y_prop[i]['linestyle'], 
				marker = self.D.y_prop[i]['marker'], 
				markevery = 1, 
				markeredgecolor = 'none', 
				label = self.D.y_prop[i]['label']
			)

		if self.D.limited_axes:
			plt.axis(self.D.axes_limits)

		# plt.fill_between(self.D.x, self.D.y[0], self.D.y[1], facecolor='yellow', alpha=0.5)

			plt.axis(self.D.axes_limits)

		# plt.fill_between(self.D.x, self.D.y[0], self.D.y[1], facecolor='yellow', alpha=0.5)

		plt.xlabel(self.D.xlabel)
		plt.ylabel(self.D.ylabel)

		plt.legend(loc='best')
	def save(self):
		self.fig.savefig(self.D.exporty['folder']+self.D.exporty['file'], format='eps')
		plt.xlabel(self.D.xlabel)
		plt.ylabel(self.D.ylabel)

		plt.legend(loc='best')



#%% Mesh

mesh = Mesh('LiwithPCM.xml')
#mesh = Mesh('LiwithPCM.xml')

n = 1  # number of refinements

for i in range(1,n):
    cell_markers = CellFunction("bool", mesh)
    for c in cells(mesh):
        if c.midpoint().x() > 0.0 and c.midpoint().x() < 0.035 - (0.005*(i-1)) and c.midpoint().y() > 0.0 and c.midpoint().y() < 0.02:
            cell_markers[c] = True
        else:
            cell_markers[c] = False
            
    mesh = refine(mesh, cell_markers)

# Define Sub-Domain
class Heatflux1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Heatflux2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < 0.001
        
class massfluxsym(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and x[1] > -0.0001) or (on_boundary and x[2] < (mt.tan(5.235833333)*x[1]+0.0001))
        
class Heatflux3(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < 0.00001

class Mappcmrho(SubDomain):
    def inside(self, x, on_boundary):
        return False #(x[0] > 0.339 and x[0] < 0.361 and x[1] > 0.019 and x[1] < 0.031) or (x[0] > 0.339 and x[0] < 0.361 and x[1] > 0.069 and x[1] < 0.081) or (x[0] > 0.039 and x[0] < 0.061 and x[1] > 0.019 and x[1] < 0.031) or (x[0] > 0.039 and x[0] < 0.061 and x[1] > 0.069 and x[1] < 0.081) or (x[0] > 0.139 and x[0] < 0.161 and x[1] > 0.019 and x[1] < 0.031) or (x[0] > 0.139 and x[0] < 0.161 and x[1] > 0.069 and x[1] < 0.081) or (x[0] > 0.239 and x[0] < 0.261 and x[1] > 0.019 and x[1] < 0.031) or (x[0] > 0.239 and x[0] < 0.261 and x[1] > 0.069 and x[1] < 0.081)

class copperPipe(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and x[0] > 0.02501 and x[0] < 0.375001) and (x[1]**2 + x[2]**2 < 0.0031**2 or x[1]**2 + (x[2]-0.02)**2 < 0.0031**2)

class copperPipe2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] > -0.00001 and x[0] < 0.0001 and x[1] < 0.00301

# Creating Mesh Function
heatflux1 = Heatflux1()
heatflux2 = Heatflux2()
heatflux3 = Heatflux3()
massfluxsym = massfluxsym() 
mappcmrho = Mappcmrho()
copper = copperPipe()
copper2 = copperPipe2()

#Initialize mesh function for boundary domains
facet_marker = FacetFunction("uint", mesh)
facet_marker.set_all(0)
#copper.mark(facet_marker,4)
heatflux1.mark(facet_marker,2)
heatflux2.mark(facet_marker,1)
copper.mark(facet_marker,4)
massfluxsym.mark(facet_marker,3)
ds = Measure("ds")[facet_marker]

#Plotting domains
#plot(facet_marker)
#File('mesh_li_var2.xml') << mesh
#interactive()

#%% Function Spaces

H = FunctionSpace(mesh, "CG", 1)        # Specific Heat Function Space
P = FunctionSpace(mesh, "CG", 1)        # pressure function space
T = FunctionSpace(mesh, "CG", 1)        # temperature function space
R = FunctionSpace(mesh, "DG", 0)        # density function space
E = FunctionSpace(mesh, "CG", 1)

eP = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
eT = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

W = FunctionSpace(mesh, MixedElement([eP, eT]))

#%% Finite Elements Problem
def forward(analysis_name,epsmet,epspcm):
    
    # Files for storing data
    cwd = oslib.getcwd()
    title = "Results_Hour-" + tm.strftime("%H") +":"+ tm.strftime("%M")+":"+ tm.strftime("%S")+"_Day-"+ tm.strftime("%d")+"-"+ tm.strftime("%m")+"-"+ tm.strftime("%y")
    outputfolder = cwd+"/"+title+"/"    
    
    file_p = File(outputfolder+analysis_name+"_P.pvd")
    file_t = File(outputfolder+analysis_name+"_T.pvd")
    file_r = File(outputfolder+analysis_name+"_R.pvd")
    file_h = File(outputfolder+analysis_name+"_H.pvd")
    file_mt = File(outputfolder+analysis_name+"_MT.pvd")
    
    file_Mt1 = File(outputfolder+analysis_name+"_Mass_Term_1.pvd")
    file_Mt2 = File(outputfolder+analysis_name+"_Mass_Term_2.pvd")
    file_Mt3 = File(outputfolder+analysis_name+"_Mass_Term_3.pvd")
    
    scriptname = oslib.path.basename(__file__)   
    copy2(cwd+'/'+scriptname,outputfolder+scriptname)
    
    # Functions

    r = Function(R, name = "Density_Cur")
    h = Function(H, name = "Heat_Flux")
    w = Function(W, name = "Mix_Sp")
    mTime = Function(H,   name = "Melting Time")
    (p, t) = split(w)
    (vM, vE) = TestFunctions(W)
    
    # Historical Variables    
    hsavep = Function(P)    # History Pressure
    hsavet = Function(T)    # History Temperature
    hsaver = Function(R)    # History 
    hsaveh = Function(H)    # History Epsilon
#    hsavemt = Function(C)
    
    hVolAdsGas = [] # Total Amount of Adsorbed Gas
    hVolGasGas = [] # Total Amount of Gaseous Gas
    hVolTotGas = [] # Total Amount of Gas
    
    hmassAdsGas = [] # Total Amount of Adsorbed Gas
    hmassGasGas = [] # Total Amount of Gaseous Gas
    hmassTotGas = [] # Total Amount of Gas
    htimes = []
    hVV = []
    hP = []
    hT = []
    hadsgen = []
    hconvecHeat = []
    hheatPCM = []
    
    hMterm1 = []
    hMterm2 = []
    hMterm3 = []
    hMterm4 = []
    hMterm5 = []
    
    hTP1 = []
    hTP2 = []
    hTP3 = []
    hTP4 = []
    hTP5 = []
    
    
    d_v = dof_to_vertex_map(H)    

    # Initial Conditions
    global p0
    p0 = Function(P, name = "Pressao - 0")
    t0 = Function(T, name = "Temperatura - 0")
    r0 = Function(R, name = "Densidade - 0")
    pin = Function(P, name = "Nietsche - 0")
    pin.assign(patm)
    h0 = Function(H, name = "Heat flux - 0")
    hSt = Function(H, name = "Storage - 1")
    hSt0 = Function(H, name = "Storage - 0")
    pguess = Function(P,   name = "Pressao - guess")
    tguess = Function(T,   name = "Temperatura - guess")
    
    Mterm1 = Function(P, name = "Termo de Massa 1")
    Mterm2 = Function(P, name = "Termo de Massa 2")
    Mterm3 = Function(P, name = "Termo de Massa 3")

    p0.assign(interpolate(Pini,P))
    t0.assign(interpolate(Tini,T))
    r0.assign(project(Qeq(Pini,Tini,epsmet,epspcm),R))
    r.assign(project(Qeq(Pini,Tini,epsmet,epspcm),R))
    h0.assign(interpolate(Constant(0.0),H))
    h.assign(interpolate(Constant(0.0),H))
    
# Variaveis para medir calor
    adsGenH = 0.0
    adsGenH0 = 0.0
    convecHeat = 0.0
    convecHeat0 = 0.0
    heatPCM = 0.0
    heatPCM0 = 0.0
    
    hSt.assign(interpolate(Constant(0.0),H))
    hSt0.assign(interpolate(Constant(0.0),H))
    inputHLim = rhopcm * Lat

    pguess.assign(p0)
    tguess.assign(t0)
    mTime.assign(interpolate(Constant(0.0), H))

    # Timestep initialization
    Cdtime = Constant(allTimes[1]-allTimes[0])
    
    dolfin.parameters["form_compiler"]["quadrature_degree"] = q_degree
    time = 0.0

    loop = 0
    timeResAnt = keyres
    
#   Variational
    Y = SpatialCoordinate(mesh)
    currFlow = Constant(0.0)
  
    n = FacetNormal(mesh)
  
#   Gradient
    theta = Constant(2.0*mt.pi)
    detJac = Y[1] # Determinant
    
    axiFact = Y[1]
    currVel = Constant(0.0)
    
    F = (
    # Massa
    inner(epsTotal(epsmet,epspcm)*(gas_rho(p,t)-gas_rho(p0,t0))/Cdtime,vM)*dx
    + inner((1-epsTotal(epsmet,epspcm))*ads_rhos0*(r-r0)/Cdtime,vM)*dx
    + inner((gas_rho(p,t)*Keff(epsmet,epspcm)/gas_mu)*grad(p),grad(vM))*dx
 
    # Neumann - mass inlet
#    + inner(axiFact*gas_rho(p,t)*(ads_K*grad(p)[0]/gas_mu),vM)*ds(1)  
    + inner(pfator*(p-pin) , vM) * ds(1)
        
    # Energy
    + inner((epsTotal(epsmet,epspcm)*gas_rho(p0,t0)*gas_cp*(t-t0)/Cdtime)*(1-epsmet)*(1-epspcm),vE)*dx
    + inner((gas_cp*ads_rhos0*(1-epsTotal(epsmet,epspcm))*(r)*(t-t0)/Cdtime)*(1-epsmet)*(1-epspcm),vE)*dx
    + inner(((1-epsTotal(epsmet,epspcm))*ads_rhos0*adsVarCp(t)*(t-t0)/Cdtime)*(1-epsmet)*(1-epspcm),vE)*dx
    + inner((Ccopp*rhocopp*(t-t0)/Cdtime)*(epsmet),vE)*dx 
    + inner((Cpcms*rhopcm*(t-t0)/Cdtime)*(epspcm),vE)*dx
    
    + inner(keff(epsmet,epspcm,t)*grad(t),grad(vE))*dx
    - inner(inner((gas_rho(p,t)*gas_cp*Keff(epsmet,epspcm)/gas_mu)*grad(p),grad(t)),vE)*dx
    + inner((1-epsTotal(epsmet,epspcm))*ads_rhos0*(DelH/gas_mg)*(r-r0)/Cdtime,vE)*dx
    - inner(epsTotal(epsmet,epspcm)*(p-p0)/Cdtime,vE)*dx
    + inner(hentrada*(t-troom) , vE) * ds(1)
    + inner(hroom*(t-troom) , vE) * ds(2)
    
    
    # Phase Change
    - inner(h,vE)*dx
#    + inner(axiFact*(t-tpc)*epspcm,vE)*dx
    )

    if rank == 0:
        print('\n---------------------------------------------------------------')
        print('\n-----------------------Starting--------------------------------')
        print('\n---------------------------------------------------------------\n')
    nts = 1
    tankVolume = assemble((interpolate(Constant(1000.0),T))*Constant(12.0)*dx)
    voladsgas = assemble((Constant(22.414)*Constant(12.0)*(r*(1-epsTotal(epsmet,epspcm))*ads_rhos0/gas_mg))*dx)
    volgasgas = assemble((Constant(22.414)*Constant(12.0)*gas_rho(p,t)*epsTotal(epsmet,epspcm)/gas_mg)*dx)
    adsmass = assemble((Constant(12.0)*ads_rhos0*(1 - epsTotal(epsmet,epspcm)))*dx)
    
    for time in allTimes:
        if nts < len(allTimes)-1:
            Cdtime.assign(allTimes[nts+1]-allTimes[nts])        

#        
        if rank == 0:
            print '------------------------------------------------------------'
            print 'Time: ' + str(time) + ' s'
            print 'Timestep: ' + str(float(Cdtime)) + ' s'
            print 'Vessel Volume: '+ str(float(tankVolume)) + ' L'
            print 'Total Gas Volume: '+ str(float(voladsgas+volgasgas)) + ' L'
            print 'Adsorbed Gas Volume: '+ str(float(voladsgas)) + ' L'
            print 'Gaseous Volume: '+ str(float(volgasgas)) + ' L'
            print 'Adsorbent Mass: '+ str(float(adsmass)) + ' kg'
            print '------------------------------------------------------------'
            
        #Nietsche
#        a = 5.95e-5
#        b = 4.56e-3
#        pin.assign(Constant((a*(time**2)+b*time+0.1)*1000000))
        
        a = 100000.0
        b = 28667.9
        c = -101.179
        d = 0.14553326474
        pin.assign(Constant((a+b*time+c*(time**2)+d*(time**3))*(1)))

#       Experimental Pressure Rise       
        bcTin = DirichletBC(W.sub(1), troom, facet_marker,1 )
        
        if time <= 2.50:
            currVel.assign(Constant(5.6128*(time/2.50)))
        else:
            currVel.assign(Constant(5.6128))
            
        bcP1 = DirichletBC(W.sub(0), patm, facet_marker, 1)
        
        bcs = []

        # Density Calculation
        assign(w.sub(0), pguess)
        assign(w.sub(1), tguess)

        h.assign(project(heatFluxLat(t0,mTime,hSt,inputHLim,epspcm),H))
#        r.assign(project((r0 + Gvar(t)*Cdtime*Qeq(p,t))/(1+Gvar(t)*Cdtime),R))
        r.assign(project(Cdtime*Gvar(t)*Qeq(p,t,epsmet,epspcm)+(Constant(1.0)-Cdtime*Gvar(t))*r0,R))
        
        ## Gateaux Derivative
        vW = TrialFunction(W)
        J = derivative(F,w,vW)
        
        # Solver Definition
        problem = NonlinearVariationalProblem(F,w,bcs,J)
        solver = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm['newton_solver']['linear_solver'] = solver_method
        prm['newton_solver']['preconditioner'] = solver_precond
        prm['newton_solver']['relative_tolerance'] = solver_reltol
        prm['newton_solver']['absolute_tolerance'] = solver_maxtol
        prm['newton_solver']['maximum_iterations'] = 25
        
        
        # Solution
        solver.solve()
        
        (p, t) = split(w)
        
        pguess.assign(project(p,P))
        tguess.assign(project(t,T))
        
        pguess.assign(pguess + (pguess-p0))
        tguess.assign(tguess + (tguess-t0))

        hSt.assign(project(phaseChangeStore(project(t,T),hSt0,hSt,inputHLim,Cdtime,mTime,epspcm),H))
        mTime.assign(project(meltingTime(hSt,project(t,T),epspcm,inputHLim),H))
        
            
        # Saving Results
        hsavep.assign(project(p,P))
        hsavet.assign(project(t,T))
        hsaver.assign(project(r,R))
        hsaveh.assign(project(hSt,H))
#            hsavemt.assign(project(mTime,C))

        
        if nts%keyres == 0:
            if rank == 0:                
                print 'Storing Results'
        # if True:
                
                
            Mterm1.assign(project((epsTotal(epsmet,epspcm)*(gas_rho(p,t)-gas_rho(p0,t0))/Cdtime),P))
            Mterm2.assign(project(((1-epsTotal(epsmet,epspcm))*ads_rhos0*(r-r0)/Cdtime),P))
            Mterm3.assign(project(project(div(-gas_rho(p,t)*Keff(epsmet,epspcm)*grad(p)/gas_mu),P),P))
        
        
            filetime = str(time).translate(None,'.')
            file_p << hsavep,htimes
            file_t << hsavet,htimes
            file_r << hsaver,htimes
            file_h << hsaveh,htimes
#                file_mt << hsavemt,htimes
            file_Mt1 << Mterm1,htimes
            file_Mt2 << Mterm2,htimes
            file_Mt3 << Mterm3,htimes

            
        timeResAnt = time%keyres

        tankVolume = assemble((interpolate(Constant(1000.0),T))*Constant(12.0)*dx)
        massadsgas = assemble(((r*(1-epsTotal(epsmet,epspcm))*Constant(12.0)*ads_rhos0))*dx )
        massgasgas = assemble((gas_rho(p,t)*epsTotal(epsmet,epspcm)*Constant(12.0))*dx)
        voladsgas = assemble((Constant(22.414/gas_mg)*Constant(12.0)*(r*(1-epsTotal(epsmet,epspcm))*ads_rhos0))*dx )
        volgasgas = assemble((Constant(22.414)*gas_rho(p,t)*Constant(12.0)*epsTotal(epsmet,epspcm)/gas_mg)*dx)
        adsGenH = adsGenH0 - assemble(((1-epsTotal(epsmet,epspcm))*Constant(12.0)*ads_rhos0*(DelH/gas_mg)*(r-r0)/Cdtime)*dx)*float(Cdtime)
        convecHeat = convecHeat0 + assemble(hroom*(t-troom)*Constant(12.0) * ds(2))*float(Cdtime)
        heatPCM = heatPCM0 + (assemble(((Cpcms*rhopcm*(t-t0)/Cdtime)*Constant(12.0)*(epspcm))*dx)*float(Cdtime)) - assemble(h*dx)*float(Cdtime)
        
        iMterm1 = assemble((epsTotal(epsmet,epspcm)*(gas_rho(p,t)-gas_rho(p0,t0))/Cdtime)*dx)
        iMterm2 = assemble(((1-epsTotal(epsmet,epspcm))*ads_rhos0*(r-r0)/Cdtime)*dx)
        iMterm3 = assemble((div(-gas_rho(p,t)*Keff(epsmet,epspcm)*grad(p)/gas_mu))*dx)
        iMterm4 = assemble((gas_rho(p,t)*(Keff(epsmet,epspcm)*grad(p)[0]/gas_mu))*ds(1))  
        iMterm5 = assemble((pfator*(p-pin)) * ds(1))

        hmassAdsGas.append(massadsgas)
        hmassGasGas.append(massgasgas)
        hmassTotGas.append(massadsgas+massgasgas)
        
        hVolAdsGas.append(voladsgas)
        hVolGasGas.append(volgasgas)
        hVolTotGas.append(voladsgas+volgasgas)
        
        hVV.append((volgasgas+voladsgas)/tankVolume)
        hadsgen.append(adsGenH)
        hconvecHeat.append(convecHeat)
        hheatPCM.append(heatPCM)
        
        hMterm1.append(iMterm1)
        hMterm2.append(iMterm2)
        hMterm3.append(iMterm3)
        hMterm4.append(iMterm4)
        hMterm5.append(iMterm5)
        
        hTP1.append(t0(tMP1))
        hTP2.append(t0(tMP2))
        hTP3.append(t0(tMP3))
        hTP4.append(t0(tMP4))
        hTP5.append(t0(tMP5))

        # Save Historical Values
        np.savetxt(outputfolder+analysis_name+"_hVolAdsGas.xml", hVolAdsGas, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_hVolGasGas.xml", hVolGasGas, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_hVolTotGas.xml", hVolTotGas, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_htimes.xml", htimes, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_hmassAdsGas.xml", hmassAdsGas, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_hmassGasGas.xml", hmassGasGas, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_hmassTotGas.xml", hmassTotGas, delimiter='\n')

        np.savetxt(outputfolder+analysis_name+"_Mass_Term1.xml", hMterm1, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_Mass_Term2.xml", hMterm2, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_Mass_Term3.xml", hMterm3, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_Mass_Term4.xml", hMterm4, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_Mass_Term5.xml", hMterm5, delimiter='\n')

        np.savetxt(outputfolder+analysis_name+"_P1.xml", hTP1, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_P2.xml", hTP2, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_P3.xml", hTP3, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_P4.xml", hTP4, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_P5.xml", hTP5, delimiter='\n')
        
        np.savetxt(outputfolder+analysis_name+"_adsGenH.xml", hadsgen, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_convecHeat.xml", hconvecHeat, delimiter='\n')
        np.savetxt(outputfolder+analysis_name+"_pcmHeat.xml", hheatPCM, delimiter='\n')

        # Updates for next timestep
        p0.assign(project(p,P))
        t0.assign(project(t,T))
        r0.assign(r)
        adsGenH0 = adsGenH
        convecHeat0 = convecHeat
        heatPCM0= heatPCM
        
#            r00.assign(r0)
        h0.assign(project(h,H))
        hSt0.assign(hSt)
        
        nts += 1
    
#    File('entlim_start.xml') << entLim
    oslib.mkdir(outputfolder +'Imagens')
    
#   Plotting Mass of gas inside the tank 
    y_prop1 = []    
    y_prop1.append({'marker':'None', 	'label':'Total Gas Volume', 	'linestyle':'-',  'color':'black'})
    y_prop1.append({'marker':'None', 	'label':'Adsorbed Gas Volume', 	'linestyle':'--',  'color':'black'})
    y_prop1.append({'marker':'None', 	'label':'Non Adsorbed Gas Volume', 	'linestyle':':',  'color':'black'})
    hvolPlot= []
    
    hvolPlot.append(np.array(hmassTotGas))
    hvolPlot.append(np.array(hmassAdsGas))
    hvolPlot.append(np.array(hmassGasGas))
    xlabel = 'Time [s]'
    ylabel = 'Mass of Gas [kg]'
    plotRes(allTimes,hvolPlot,y_prop1,xlabel,ylabel, outputfolder +'Imagens',"/Gas_Mass.eps")
    
#   Plotting Volume of gas inside the tank 
    y_prop1 = []    
    y_prop1.append({'marker':'None', 	'label':'Total Gas Volume', 	'linestyle':'-',  'color':'black'})
    y_prop1.append({'marker':'None', 	'label':'Adsorbed Gas Volume', 	'linestyle':'--',  'color':'black'})
    y_prop1.append({'marker':'None', 	'label':'Non Adsorbed Gas Volume', 	'linestyle':':',  'color':'black'})
    hvolPlot= []
    
    hvolPlot.append(np.array(hVolTotGas))
    hvolPlot.append(np.array(hVolAdsGas))
    hvolPlot.append(np.array(hVolGasGas))
    xlabel = 'Time [s]'
    ylabel = 'Amount of Gas [L]'
    plotRes(allTimes,hvolPlot,y_prop1,xlabel,ylabel, outputfolder +'Imagens',"/Gas_Volume.eps")

#   Plotting Storage Capacity
    
    y_prop1 = []    
    y_prop1.append({'marker':'None', 	'label':'Total Gas Volume', 	'linestyle':'-',  'color':'black'})
    hvolPlot= []
    
    hvolPlot.append(np.array(hVV))
    xlabel = 'Time [s]'
    ylabel = 'Storage Capacity [V/V]'
    plotRes(allTimes,hvolPlot,y_prop1,xlabel,ylabel, outputfolder +'Imagens',"/Storage Capacity.eps")
    
#   Plotting Volume of gas inside the tank 
    y_prop1 = []    
    y_prop1.append({'marker':'None', 	'label':'Term 1', 	'linestyle':'-',  'color':'black'})
    y_prop1.append({'marker':'None', 	'label':'Term 2', 	'linestyle':'-',  'color':'red'})
    y_prop1.append({'marker':'None', 	'label':'Term 3', 	'linestyle':'-',  'color':'green'})
    y_prop1.append({'marker':'None', 	'label':'Term 4', 	'linestyle':'-',  'color':'blue'})
    y_prop1.append({'marker':'None', 	'label':'Term 5', 	'linestyle':'-',  'color':'yellow'})    
    hvolPlot= []
    
    hvolPlot.append(np.array(hMterm1))
    hvolPlot.append(np.array(hMterm2))
    hvolPlot.append(np.array(hMterm3))
    hvolPlot.append(np.array(hMterm4))
    hvolPlot.append(np.array(hMterm5))
    xlabel = 'Time [s]'
    ylabel = 'Mass Variation [kg/s]'
    plotRes(allTimes,hvolPlot,y_prop1,xlabel,ylabel, outputfolder +'Imagens',"/Termos_Massa.eps")

#   Plotting Temperatures inside the tank 
    y_prop1 = []    
    y_prop1.append({'marker':'None', 	'label':'P1 ('+ str(tMP1.x())+','+str(tMP1.y())+')', 	'linestyle':'-',  'color':'black'})
    y_prop1.append({'marker':'None', 	'label':'P2 ('+ str(tMP2.x())+','+str(tMP2.y())+')', 	'linestyle':'-',  'color':'red'})
    y_prop1.append({'marker':'None', 	'label':'P3 ('+ str(tMP3.x())+','+str(tMP3.y())+')', 	'linestyle':'-',  'color':'green'})
    y_prop1.append({'marker':'None', 	'label':'P4 ('+ str(tMP4.x())+','+str(tMP4.y())+')', 	'linestyle':'-',  'color':'blue'})
    y_prop1.append({'marker':'None', 	'label':'P5 ('+ str(tMP5.x())+','+str(tMP5.y())+')', 	'linestyle':'-',  'color':'yellow'})    
    hvolPlot= []
    
    hvolPlot.append(np.array(hTP1))
    hvolPlot.append(np.array(hTP2))
    hvolPlot.append(np.array(hTP3))
    hvolPlot.append(np.array(hTP4))
    hvolPlot.append(np.array(hTP5))
    xlabel = 'Time [s]'
    ylabel = 'Temperature [K]'
    plotRes(allTimes,hvolPlot,y_prop1,xlabel,ylabel, outputfolder +'Imagens',"/Temperatura_Pontos.eps")
    
#   Plotting Head Balance 
    y_prop1 = []    
    y_prop1.append({'marker':'None', 	'label':'Generated Adsorption Heat', 	'linestyle':'-',  'color':'black'})
    y_prop1.append({'marker':'None', 	'label':'Heat Dissipaded by Convection', 	'linestyle':'-',  'color':'red'})
    y_prop1.append({'marker':'None', 	'label':'Energy Stored in PCM', 	'linestyle':'-',  'color':'green'})
    
    hvolPlot= []
    
    hvolPlot.append(np.array(hadsgen))
    hvolPlot.append(np.array(hconvecHeat))
    hvolPlot.append(np.array(hheatPCM))
    xlabel = 'Time [s]'
    ylabel = 'Total Heat [J]'
    plotRes(allTimes,hvolPlot,y_prop1,xlabel,ylabel, outputfolder +'Imagens',"/Balanco_Calor.eps")

#   Processing Position
    Pos = FunctionSpace(mesh, "CG", 2)
    
#   Plotting Pressure
#   Plotting Temperature
    return p0,t0,r0 #,timeopt

# MAIN
#epsc = interpolate(copperMap(degree=1),E)
#plot(epspcm)
#interactive()
#epspcm = Function(E,outputfolder+"Optimizating_adsorb_5"+"/"+"PCM"+"_epsopt.xml",name = "PCM_Dist",  )

#epsmet = interpolate(Constant(0.0),E)
epsmet = interpolate(copperMap(degree=1),E)

#epspcm = interpolate(Constant(0.0),E)
epspcm = interpolate(Pcmmap(degree=1),E)

p,t,r = forward('Resultado_Adsorcao_comPCM_finer',epsmet, epspcm)

# Uncomment to restart
#File('p_start.xml') << p
#File('r_start.xml') << r
#File('t_start.xml') << t

'''
if __name__ == "__main__":
    epspcm = Function(E,name = "PCM_Dist", annotate = True)
#    epspcm = Function(E,model.outputfolder+"Optimizating_3p5_"+"/"+"PCM"+"_epsopt.xml",name = "PCM_Dist", annotate = True)
    eps = Function(E, name = "Porosity", annotate = True)
    eps.assign(interpolate(Constant(1.0),E))
#    epspcm = interpolate(model.Pcmmap(),E)
    epspcm.assign(interpolate(Constant(0.2),E))   
    p, t, r= forward(eps,epspcm,'Optimizating_3p5_','PCM',True,1)
    
    # for the equations recorded on the forward run
    adj_html(model.outputfolder+'Optimizating_3p5_'+'/'+'PCM'+"forward.html", "forward")
    # for the equations to be assembled on the adjoint run
    adj_html(model.outputfolder+'Optimizating_3p5_'+'/'+'PCM'+"adjoint.html", "adjoint")  
    
#    parameters["adjoint"]["stop_annotating"] = True

    controls = File(model.outputfolder+'Optimizating_3p5_'+'/'+'PCM'+"control_iterations.pvd")
    a_viz = Function(E, name="ControlVisualisation")
    ob_viz = []
    def eval_cb(j, a):
        a_viz.assign(a)
        ob_viz.append(j)
        controls << a_viz
        plt.plot(ob_viz)
        plot(a_viz, title = "opt", key = "Opt", input_keys="r")   
        numpy.savetxt(model.outputfolder+"Optimizating_3p5_"+"/"+"PCM"+"_Objective_Func."+".xml", ob_viz, delimiter='\n')
        File(model.outputfolder+"Optimizating_3p5_"+"/"+"PCM"+"_epsopt.xml") << a_viz
    
    J = Functional(-r*dx*dt[FINISH_TIME])
    m = Control(epspcm)
    
    # Bound constraints
    lb = project(Constant(0.0),E)
    ub = project(Constant(1.0),E)
    
#    dJdnu = compute_gradient(J, m)
          
    rf = ReducedFunctional(J, m, eval_cb_post = eval_cb)
    
    problem = MinimizationProblem(rf,  bounds=(lb,ub))
    parameters = { 'maximum_iterations': 20 }
    
    
    solver = IPOPTSolver(problem, parameters=parameters)
    Eps_het_opt = solver.solve()
'''