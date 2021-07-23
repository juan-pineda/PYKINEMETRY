#!/usr/bin/env python
"""

Scientific Data Analysis of Galaxy Velocity Maps. Part 2: Obtaining synthetic models


    Part 2: To determine the reliability of the studied model we worked with synthetic models,
    these are generated based on velocity parameters of real galaxies, once these models were
    obtained we proceeded to perturb them with a representative sample of debris obtained from
    the adjustment of real data obtained from G. Bekiaris \cite{bekiaris2016kinematic} and from
    the data of several studies of galaxies by \href{https://cesam.lam.fr/fabryperot/}{PerotFabry}.


Authors:
        José Miguel Ladino <jmladinom@unal.edu.co>, Universidad Nacional de Colombia
        Omar Asto Rojas <oasto1310@gmail.com>, Uiversidad Nacional de Ingenería, Perú
        Jennifer Grisales <jennifer.grisales@saber.uis.edu.co>, Universidad Industrial de Santander, Colombia
Please contact us in case of questions or bugs.

Supervisor:
        PhD. Juan Carlos Basto Pineda

This code is the core of the Data Analysis final course project inside LA-CoNGA Physics program
2021

Please import the following libraries:

import numpy as np
import matplotlib as mt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import leastsq
import scipy.stats as stats
from astropy.io import fits
import scipy.optimize as optimize
import pandas as pd
from matplotlib.pyplot import figure
import seaborn as sb
from numpy import random
import pathlib
"""

import numpy as np
import matplotlib as mt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import leastsq
import scipy.stats as stats
from astropy.io import fits
import scipy.optimize as optimize
import pandas as pd
from matplotlib.pyplot import figure
import seaborn as sb
from numpy import random
import pathlib

### Definición de la función de velocidad y error
def Funcion(xx , yy, p):
	
	

	#p = [x0, y0, v_sys, i, phi_0, V_t, R_0, a, g]   
	R = np.sqrt((xx-p[0])**2 + (yy-p[1])**2)
	def psi(a , b):
		psi =  np.rad2deg(np.arctan2((b - p[1]) , a - p[0] )) + 180
		return psi 
	zz = psi(xx, yy)
	phi = np.rot90(zz,3) 
    
	def alpha2(phi, phi_0, i):
		alpha=np.sqrt((np.cos(np.deg2rad(phi-phi_0)))**2+(np.sin(np.deg2rad(phi-phi_0)))**2/(np.cos(i*np.pi/180))**2)
		return alpha
	alpha1 = alpha2(phi, p[4], p[3])
  
	r=alpha1*R

	v_c = p[5] * ((r/p[6])**(p[7]))/(1 + (r/p[6])**(p[8]))
	V_los = p[2] + v_c*((np.sin(np.deg2rad(p[3])))*np.cos(np.deg2rad(phi - p[4])))/alpha1
    
	return V_los
def error(tpl, x, y, z):
	mask = np.isnan(z)   
	E =  Funcion(x, y, tpl) - z
    
	return np.ravel(E[~mask])

## Primera función


def Parametros_(name_path):
	"""
    This function lets us to obtain from the kinetic parameters of real galaxies.
    
	For the function, the path to the parameters is given, these data are in .csv format.
	The kinetic parameters were taken from the paper "Kinematic Modelling of disc galaxies
    using graphics processing units" by G.Bekiaris
	
	"""

	p = pathlib.Path(name_path)
	Z22 = []
	for f in p.glob('*.csv'):
		Z = pd.read_csv(f, header = 0)
		Z = np.array(Z, dtype = np.float64 )
		Z22.append(Z)
    
	A = np.array(Z22)
    
    
	Z2 = A[0,:,:]
	## Adición de parámetros
	V_sys = np.append(Z2[:, 0], Z2[:, 1])
	PA = np.append(Z2[:, 2], Z2[:, 3])
	i = np.append(Z2[:, 4], Z2[:, 5])

	x_0 = 4*random.rand(346) + 78
	y_0 = 4*random.rand(346) + 78

	R_0 = 10*random.rand(346) + 5
	V_t = 200*random.rand(346) + 50
	a = random.rand(346) + 0.5
	g = random.rand(346) + 0.5

	Params_Bek = np.vstack([x_0, y_0, V_sys, i, PA, V_t, R_0, a, g]).T

	p = ["x0", "y0", "v_sys", "i", "phi_0", "V_t", "R_0", "a", "g"] 
	for j in range(len(Params_Bek.T)):
		sb.displot((Params_Bek.T[j]).ravel(), color='#F2AB6D', bins=10, kde=False)
		plt.xlabel(p[j])
		plt.ylabel('Cuentas')
		plt.show()

	return Params_Bek

### Segunda Función:  Mejor ajuste de los parámetros

def Galaxies(name_path):
	"""
    This function gives us an array of velocity matrices with real data.
    
	For the function the path where the parameters are given, this data is put in .fits format:
	"""
	p = pathlib.Path(name_path)
	galaxies = []
	for f in p.glob('*.fits'):
		hdu = fits.open(f)
		Z1= hdu[0].data
		galaxies.append(Z1)
    
	return galaxies


## Para esta función primero se utiliza la función Galaxies, que nos da un arreglo de matrices de velocidad con datos reales


def Params_ajus(matriz_de_velocidad, matriz_de_par):
	"""
    For this function we first use the Galaxies function, which gives us
    an array of velocity matrices with real data.
    
    Two parameters are given, first the set of velocity matrices with real
    data and then the parameter matrix and then the kinetic parameter 
    matrix obtained with the 'Parametros_' function.
    
	Observación
    -----------
    El problema que se encontró en dicho array es que los paŕametros que se
    completaron, no nocesariamente son apegados a la realidad, que es muy 
    importante en el análisis a través de modelos sintéticos.
    
    Para solucionar este problema, se procedió a ajustar los parámetros a galaxias
    obtenidas la base de datos [Fabry Perot Website](https://cesam.lam.fr/fabryperot/)
    en donde se pueden obtener matrices de velocidad de diferentes galaxias.
    Para el análisis se busco las galaxias pertenecientes al Catálogo General
    Uppsala (UGC, por sus siglas en inglés) y que se hayan estudiado en el 
    paper referido para la obteción de parámetros cinéticos.
    
    Parameters
    ----------
    matriz_de_velocidad : 2D Array
                          the set of velocity matrices with real data and 
                          then the parameter matrix.
    matriz_de_par : 2D Array
                    the kinetic parameter matrix obtained with the 
                    'Parametros_' function.
    return
    ------
    Params_Bek1 : np.array
    
	"""
	Best_params = []
	X = []
	Y = []

	for i in range(len(matriz_de_par)):
        
		Z = random.choice(matriz_de_velocidad)
		x = np.arange(0,len(Z))
		y = np.arange(0,len(Z))
		Y1, X1 = np.meshgrid(x,y)

		best = leastsq(error, matriz_de_par[i,:], args= (X1, Y1, Z), full_output=1)
		Best_params.append(best[0])

	G = np.array(Best_params)

	E  = Parametros_("../op8_scientific_data_analysis_of_galaxy_velocity_maps/datos_fabry_GHASP/")

	Params_Bek1 = np.vstack([E[:,0], E[:,1], E[:,2], E[:,3], E[:,4], E[:,5], G[:,6], G[:,7], G[:,8]]).T

	p = ["x0", "y0", "v_sys", "i", "phi_0", "V_t", "R_0", "a", "g"]

	for j in range(3):
		sb.displot((Params_Bek1.T[j+6]).ravel(), color='#F2AB6D', bins=25, kde=False)
		plt.xlabel(p[j+6])
		plt.ylabel('Cuentas')
		plt.show()

	return Params_Bek1

### Tercera Función: Esta función escoge valores aleatorios del arreglo de parámetros

def choice(values, n):

	"""
	Function that generates 1000 random kinetic parameters
    for the generation of synthetic models.
    
    Parameters
    ----------
    As input, the matrix of the obtained parameters and the
    number of synthetic models you want to obtain are given.
    
    values : Params_Bek
    n : number of synthetic models you want to obtain are given
    
    Returns
    -------
    g : integers
        set of parameters
    
	"""
	g = []
	j = 0
	values1 = values.T
	for j in range(0,n):
        
		v = []
		for i in range(len(values1)):
			v.append(random.choice(values1[i,:]))
		v = np.array(v)
		g.append(v)
	g = np.array(g)
	return g

def Funcion1(p):

	x1 = np.arange(0,160)
	y1 = np.arange(0,160)
	yy, xx = np.meshgrid(x1,y1)
	
	

	#p = [x0, y0, v_sys, i, phi_0, V_t, R_0, a, g]   
	R = np.sqrt((xx-p[0])**2 + (yy-p[1])**2)
	def psi(a , b):
		psi =  np.rad2deg(np.arctan2((b - p[1]) , a - p[0] )) + 180
		return psi 
	zz = psi(xx, yy)
	phi = np.rot90(zz,3) 
    
	def alpha2(phi, phi_0, i):
		alpha=np.sqrt((np.cos(np.deg2rad(phi-phi_0)))**2+(np.sin(np.deg2rad(phi-phi_0)))**2/(np.cos(i*np.pi/180))**2)
		return alpha
	alpha11 = alpha2(phi, p[4], p[3])
  
	r=alpha11*R

	v_c = p[5] * ((r/p[6])**(p[7]))/(1 + (r/p[6])**(p[8]))
	V_los = p[2] + v_c*((np.sin(np.deg2rad(p[3])))*np.cos(np.deg2rad(phi - p[4])))/alpha11
    
	return V_los


def mode_sinte(g):
    """
    This function creates n synthetic models from set of parameters 'g'
    
    Parameters
    ----------
    g : set of parameters
    
    Returns
    -------
    l : array
        n velocity synthetic models 
    """
	l = []
	for i in range(len(g)):
		j = g[i,:]
		l.append(Funcion1(j))
	l = np.array(l)
	return l


### Cuarta Función: Hallando la muestra de errores

def Error_muestra(matriz_de_velocidad):
	"""
	Todos los datos reales obtenidos de [Fabry Perot Website](https://cesam.lam.fr/fabryperot/) se 
	ajustan con la función de Epinat estudiada en la primera parte, para obtener una muestra de restos.
	"""
	matriz_de_velocidad = np.array(matriz_de_velocidad)
	Best_params = []
	F = []
	R = []
    
	for i in range(len(matriz_de_velocidad)):
        
		p = [len(matriz_de_velocidad[i])/2,len(matriz_de_velocidad[i])/2,5,30,-90,350,10,0.9,1.1] 
		x = np.arange(0,len(matriz_de_velocidad[i]))
		y = np.arange(0,len(matriz_de_velocidad[i]))
		Y1, X1 = np.meshgrid(x,y)

		best = leastsq(error, p, args= (X1, Y1, matriz_de_velocidad[i]), full_output=1)
		Best_params.append(best[0])

		F1 = Funcion(X1, Y1, Best_params[i])
		F.append(F1)
		r = matriz_de_velocidad[i] - F[i]
		R.append(r)
        
	### Obteniendo restos de 160x160
	Restos = []
	for j in range(len(R)):
		e = R[j]
		e[np.isnan(e)] = 0
            
		d = []
		if len(e) < 200:
			z = np.zeros((200, 200))
			t = np.resize(e,z.shape)
			c = t + z
			for i in range(160):
				f = []
				for k in range(160):
 					f.append(c[i+20,k+20])
				d.append(f)
		else:                
			for i in range(160):
				f = []
				for k in range(160):
					f.append(e[i+20,k+20])
				d.append(f)
		Restos.append(d)
		#np.array(Restos)    
	return Restos

## Quinta Función: Determinación de la incertidumbre de cada parámetro a través de la generación de modelos sintéticos

def parametros_incer(Datos_sint, Restos,n):
    """
    Once the sample of residuals and the synthetic models have been obtained,
    a synthetic model is taken and an error is added to it from the error sample
    to obtain a perturbed model and add an error from the error sample to
    obtain a perturbed model. Once the perturbed model is obtained, it is
    adjusted with the Epinat function studied, the new adjusted parameters 
    are matched to the parameters of the synthetic parameters of the unperturbed
    synthetic model. It is noted that if there is a difference it is due to the
    sum of the remainder, there are times when the same parameter values are 
    obtained and it is because there are residuals that are very small.
    
    All obtained are added up and divided by the total number of times the characteristic
    residuals were taken, in order to were taken, in order to obtain an overall
    uncertainty related to the generation of synthetic models obtained from real
    data. 
    
    synthetic models obtained from real data.
    """
	x = np.arange(0,999)
	y = np.arange(0,17)
    
	r1 = random.choice(x)
	r2 = random.choice(x)
	incer = []
    
	x1 = np.arange(0,160)
	y1 = np.arange(0,160)
	Y, X = np.meshgrid(x1,y1)
	for i in range(n):
		r1 = random.choice(x)
		r2 = random.choice(y)
        
		Z = Datos_sint[r1,:,:]
        

		p = [80,80,5,30,-90,350,10,0.9,1.1] 
		best = leastsq(error, p, args= (X, Y, Z), full_output=1)
        
        
		Z1 = np.array(Restos[r2]) + Datos_sint[r1,:,:]
        

		best1 = leastsq(error, p, args= (X, Y, Z1), full_output=1)
        
		B = best1[0] - best[0]
        
		incer.append(B)
    
	incer_final = []
	for j in range(len(incer)-1):
		sumas = (np.abs(incer[j]) + np.abs(incer[j+1]))/n    
		incer_final.append(sumas)
    
	incer_final = np.array(incer_final)

	p = ["x0", "y0", "v_sys", "i", "phi_0", "V_t", "R_0", "a", "g"]
	
	
	
	for i in range(len(p)):	 
	
		print("la incertidumber de", p[i], "es:", incer_final[0,i])

	return incer_final    
