# -*- encoding: latin1 -*-

"""-------------------------------------
DARK SPIDER,                __   __  
The 2D Velocity Maps Fit   (  \,/  ) 
                            \_ | _/  
Butterfly Software System   (_/ \_)  
  Juan Pineda, Fabricio Ferrari 2011	             
-------------------------------------

Updated: February 1, 2013
"""


"""
Full documentation at:
"""

import numpy as np


global	eps
eps	= 1e-300

print __doc__


# This class will perform different actions related to 2D velocity maps.
class Maps(object):

	# Input variables are expected to be in the following units:
	# Vel_map	= [km/s]
	# pgeo		= [psi0,i0,x0,y0,Vsys] = [deg,deg,pix,pix,km/s]
	# Take care of adding 90 deg to 'psi0' before if necessary (case of GHASP)
	# beta		= [deg/pix] (plate scale)
	# Err_map	= [km/s]
	def __init__(self,Vel_map,p_geo,beta=1,Err_map=1):
		"""Initializing global variables and setting useful quantities up.
		"""
                self.Vp                                 = Vel_map
                self.p_geo                              = p_geo

		self.psi0				= np.radians(p_geo[0])
		self.i0					= np.radians(p_geo[1])
		# Note that since now on self.psi0 and self.i0 will be in radians
		self.x0					= p_geo[2]
		self.y0					= p_geo[3]
		self.Vsys				= p_geo[4]

                self.sin_i                              = np.sin(self.i0)
                self.cos_i                              = np.cos(self.i0)
		self.dim				= self.Vp.shape
		self.index				= np.where(~np.isnan(self.Vp))
		self.index_nan				= np.where(np.isnan(self.Vp))
		# Number of not 'NaN' values.
		self.N					= self.index[0].size
		
                if type(Err_map) == float:
                        self.sigmaVp                    = Err_map*np.ones(self.dim)
                        self.sigmaVp[self.index_nan]    = np.NaN
                else:
                        self.sigmaVp                    = Err_map

                # Fundamental geometrical quantities.
		self.r,self.alpha,self.psi		= self.Geometry(self.p_geo)
		self.arcsec				= self.r*self.beta*3600.
		
                # Circular velocity, deprojected from the observed velocity
                self.Vc					= self.deproject(self.Vp)
                self.sigmaVc                            = self.sigmaVp * self.alpha / (self.sin_i * np.cos(self.psi-self.psi0))

	# Fundamental geometrical quantities. 'r' is in pixels (plane of the galaxy),'alpha','psi' in radians.
	def Geometry(self,p_geo):
		"""Given geometrical parameters, this module computes the auxiliary matrices r,alpha,psi.
		"""
		psi0,i0,x0,y0,Vsys	= p_geo[:]
		y,x			= np.mgrid[:self.dim[0],:self.dim[1]]
		R			= np.sqrt((x-x0)**2. + (y-y0)**2.)
		psi			= np.arctan2(y-y0, x-x0)
		alpha			= np.sqrt((np.cos(psi-psi0))**2 + (np.sin(psi-psi0))**2/self.cos_i**2)
		r			= alpha*R
		return r,alpha,psi

	# Calculates the velocity projected along the line-of-sight.
	def project(self,Vc):
		Vp			= self.Vsys + Vc * self.sin_i * np.cos(self.psi-self.psi0) / self.alpha
		return Vp

	# Calculates the circular velocity corresponding to a measured projected velocity.
	def deproject(self,Vp):
                Vc                      = (Vp-self.Vsys) * self.alpha / (self.sin_i * np.cos(self.psi-self.psi0))
		return Vc

	# This modul sort a lot of velocity points according to their radius.
	# Inputs should be same-size matrices for radius, Velocity and Errors. An index tuple is also required.
	# If there is no error map to be sorted, set that input as None.
	def sort_curve(self,r_map,Vel_map,Err_map,index):
		hh			= np.argsort(np.ravel(r_map[index]))
		if type(Err_map)!=None:
			srt_curve	= np.zeros([r_map[index].size,3])
			srt_curve[:,2]	= np.ravel(Err_map[index])[hh[:]]
		else:
			srt_curve	= np.zeros([r_map[index].size,2])
		srt_curve[:,0]		= np.ravel(r_map[index])[hh[:]]
		srt_curve[:,1]		= np.ravel(Vel_map[index])[hh[:]]
		return srt_curve

	# Creates a rotation curve (cloudy), de-projecting all the projected velocities in the 2D map.
	# It excludes an angular region around the minor axis given by 'angle' [deg].
	def filter_curve(self,angle):
		angle			= np.radian(angle)
		mask			= np.ones(self.dim)
		mask[self.index_nan]	= np.nan
                # !!! En lugar de self.teta utilizar (psi-psi0)
		index			= np.where(np.abs(self.teta*mask - np.pi/2) > angle)
		index_red		= np.where((self.teta*mask - np.pi/2) > angle)
		index_blue		= np.where((np.pi/2-self.teta*mask) > angle)
		filterrc		= self.sort_curve(self.arcsec,self.Vc,self.sigmaVc,index)
		appr			= self.sort_curve(self.arcsec,self.Vc,self.sigmaVc,index_blue)
		receed			= self.sort_curve(self.arcsec,self.Vc,self.sigmaVc,index_red)
		# Those output velocity curves will be in [km/s],[arcsec]
		return filterrc,appr,receed


	######################## if a.header['pixelsize'] <= 2 arcsec:
    ##################################     vel = a.velmap[:][mitad-1:mitad+2]
    ################################## else:
    ##################################     vel = a.velmap[:][mitad]
    ################################## radius = np.abs (np.arange(N) - mitad)
    ################################## nan_mask = numpy.isnan(vel)
    ################################## vel = vel[~nan_mask]
    ################################## radius = radius[~nan_mask] 
    ################################## sin_incl = np.sin(np.radians(incli))
    ################################## vel = np.abs(vel / sin_incl)
    ################################## # Aquí hay que tener cuidado con las unidades!
    ################################## # Hay que usar a.header['SPAT_RES'] en unidades de pixeles!
    ################################## width = a.header['SPAT_RES'] 
    ################################## bins = np.arange(0,np.max(radius)+width,width)
    ################################## belong_bins = np.digitize(radius,bins)
    ################################## df = pd.DataFrame({'bin':belong_bins, 'vel':vel})
    ################################## vel_mean = df.groupby('bin').apply(lambda x: np.mean(x.vel))
    ################################## vel_std = df.groupby('bin').apply(lambda x: np.std(x.vel))
    ################################## # En qué unidades queremos retornar R?
    ################################## R = width/2 + np.arange(0, np.max(belong_bins)-1, 1)*width

