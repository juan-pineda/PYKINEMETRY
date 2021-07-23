# -*- encoding: latin1 -*-

"""-------------------------------------
SPIDER,                     __   __  
Velocity Maps Analysis     (  \,/  ) 
                            \_ | _/  
Contributors                (_/ \_)  
  Juan Pineda,
  Juan Manuel Pacheco,
  Fabricio Ferrari	             
-------------------------------------

Updated: June 3, 2021
"""


"""
Full documentation at:
"""

"""
This class provides a  framework to analyse 2D velocity maps of galaxies, with
the final aim of extracting fair rotation curves with different methods.
"""
import numpy as np


global	eps
eps	= 1e-300

print __doc__


class Maps(object):
    """
    This class will perform different actions related to 2D velocity maps.
    It provides methods to calculate intermediate geometrical quantities,
    and to compute the rotation curve by applying eliptical binning on the
    2D velocity map.

    Input variables:
   
    Vel_map:
        2D array with the observed velocity map in [km s^-1]
    pgeo:
       A list of five numbers defining the geometrical orientation of the disk
       pgeo = [psi0,i0,x0,y0,Vsys]
       - psi0: position angle of the major axis [deg]
       - i0: inclination of the disk in the sky [deg]
       - x0: row coordinate of the center of rotation [pix]
       - y0: column coordinate of the center of rotation [pix]
       - Vsys: Systemic velocity of the galaxy [km s^-1]

    Warning:
        In some cases it is necessary to add 90 deg to 'psi0' to set the right 
        frame of coordinates, it depends on the definition os the value of Psi0
        reported for the galaxy.

    beta
        float, pixel plate scale [deg/pix] or [arcsec/pix] or [kpc/pix]
        It is set to 1 by default, giving the radii in pixels in that case
    Err_map:
        float or 2D array
        Measuremente errors in the projected velocities [km s^-1]
        If float, it assumes the same velocity error in all measurements

    """

    def __init__(self,Vel_map,p_geo,beta=1,Err_map=1):
        """
        Initializing instance attributes and setting useful quantities up.
        The necessary parameters to instatiate the class are described in the
        docstring of the definition of the class itself.

        """
        # Velocity map
        self.Vp = Vel_map
        
        # List with the geometricla parameters
        self.p_geo = p_geo

        # conversion factor from 
        self.beta = beta

        # Extract the geometrical parameters individually
        # Since now on self.psi0 and self.i0 will be in radians
        self.psi0 = np.radians(p_geo[0])
        self.i0	= np.radians(p_geo[1])
        self.x0	= p_geo[2]
        self.y0	= p_geo[3]
        self.Vsys = p_geo[4]

        # Compute trigonometric functions as they are constantly needed
        self.sin_i = np.sin(self.i0)
        self.cos_i = np.cos(self.i0)

        # Dimensions of the velocity map. Main arrays will have same dimensions
        self.dim = self.Vp.shape

        # Number of valid pixels, and where possible NaN's are
        self.index = np.where(~np.isnan(self.Vp))
        self.index_nan = np.where(np.isnan(self.Vp))
        # Number of not 'NaN' values.
        self.N = self.index[0].size

        # If the error is a float number, an error map is created replicating it 
        if isinstance(Err_map,float):
            self.sigmaVp = Err_map*np.ones(self.dim)
            self.sigmaVp[self.index_nan] = np.NaN
        else:
            self.sigmaVp = Err_map

        # create the auxiliary matrices self.r, self.alpha, self.psi
        self.Geometry()


    def Geometry(self):
        """
        Given geometrical parameters, this module computes the auxiliary matrices r,alpha,psi.

        Parameters
        ----------
        p_geo: list
            A list with the 5 essential geometrical parameters

        Returns
        -------
        self.r: np.narray
            Matrix of true radii to the center in the plane of the disk
        self.alpha:
            Auxiliar matrix for distance transformations
        self.psi:
            Matrix of angular positions with respect to the major axis
        """
        y,x = np.mgrid[:self.dim[0],:self.dim[1]]
        R = np.sqrt((x-self.x0)**2. + (y-self.y0)**2.)
        self.psi = np.arctan2(y-self.y0,self.x0-x) + np.pi
        self.psi = np.where(self.psi>=self.psi0, self.psi-self.psi0, 2*np.pi + self.psi-self.psi0)
        self.cos_psi = np.cos(self.psi)
        self.sin_psi = np.sin(self.psi)
        self.alpha = np.sqrt((self.cos_psi)**2 + (self.sin_psi)**2/self.cos_i**2)
        self.r = self.alpha*R
        # convert to angular units if specified
        self.r = self.r*self.beta


    def deproject(self):
        """
        Calculates the circular velocity map by deprojecting the observed one
        
        self.Vc:
            2D array containing the circular velocity associated to every pixel
        self.sigma_Vc:
            2D array of errors in the circular velocities
        """
        factor = self.alpha / (self.sin_i * self.cos_psi)
        self.Vc = (self.Vp-self.Vsys) * factor
        self.sigma_Vc = self.sigma_Vp * factor


    def cloudy_rotation_curve(self,angle):
        """
        Creates a rotation curve de-projecting all individual velocity pixels.
        It excludes a given angular region around the minor axis
        Input:
        angle:
            float, angular width to be ignored on each side of minor axis [deg]
        Output:
        filterrc:
            array of dimensions Nx3, being N the number of valid pixels
            columns are [radius, Vcir, Error_Vcir], ordered by increasing radius
        appr:
            Same as filterrc, but only for the approaching side
        receed:
            Same as filterrc, but only for the receeding side
        """
        angle = np.radians(angle)
        
        # angular region to exclude around to 'positive minor axis'
        to_reject_1 = (np.pi/2-angle <= self.psi <= np.pi/2+angle)
        # angular region to exclude around to 'negative minor axis'
        to_reject_2 = (3*np.pi/2-angle <= self.psi <= 3*np.pi/2+angle)
        # index of all pixels in the allowed region
        index_all = ~(to_rect_1 | to_reject_2)

        # index of the approaching (blue) and the receeding (red) sides
        index_red = (np.pi/2+angle < self.psi < 3*np.pi/2-angle)
        index_blue = (np.pi/2+angle > self.psi) | (3*np.pi/2+angle < self.psi)

        # check which is the truly approaching/receeding side 
        # (because the disk may rotate in one direction or the other)
        # The receeding side must be associated to 'red' due to redshift
        # This means larger values because receeding velocities are positive
        V_red = np.nanmean(self.Vp[index_red])
        V_blue = np.nanmean(self.Vp[index_blue])
        if V_blue > V_red:
            tmp = index_red.copy()
            index_red = index_blue.copy()
            index_blue =tmp

        filterrc = self.sort_curve(self.r,self.Vc,self.sigmaVc,index_all)
        appr = self.sort_curve(self.r,self.Vc,self.sigmaVc,index_blue)
        receed = self.sort_curve(self.r,self.Vc,self.sigmaVc,index_red)
        # Those output velocity curves will be in [km/s]
        return filterrc,appr,receed


    def sort_curve(self,r_map,Vel_map,Err_map,index):
        """
        This module sort a lot of velocity points according to their radius.
        Inputs should be same-size matrices for radius, Velocity and Errors. An index tuple is also required.
        If there is no error map to be sorted, set that input as None.
        """
        hh                      = np.argsort(np.ravel(r_map[index]))
        if type(Err_map)!=None:
            srt_curve   = np.zeros([r_map[index].size,3])
            srt_curve[:,2]      = np.ravel(Err_map[index])[hh[:]]
        else:
            srt_curve   = np.zeros([r_map[index].size,2])
            srt_curve[:,0]              = np.ravel(r_map[index])[hh[:]]
            srt_curve[:,1]              = np.ravel(Vel_map[index])[hh[:]]
        return srt_curve



    # Falta el binning -> considerar el pesado con las luminosidades (cuando hay barras de error, eso ya debió jugar su papel)

#    def long_slit_curve(self,ncol):
#        N = self.dim[0]
#        center = (N-1)//2
#    if ncol = 1:
#        vel = self.Vp[:][center]
#    else:
#        nn = ncol//2
#        vel = self.Vp[:][center-nn:center+nn+1]


    ################################## if a.header['pixelsize'] <= 2 arcsec:
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

