"""

==============
VelocityMaps
==============

This class provides a  framework to analyse 2D velocity maps of galaxies, with
the final aim of extracting fair rotation curves with different methods.

The most important equation to keep in mind is, using the polar angle theta in
the plane of the galaxy:

    Vp = Vsys + Vc * sin(i) * cos(theta)

Which is equivalent to the following one, using the polar angle Psi in the plane
of the sky:

    Vp = Vsys + Vc * sin(i) * cos(Psi) / alpha

Notes
-----
- Python representation of images/matrices reverses the vertical axis with respect
to the external frame of references. The GHASP velocity maps set the 0 of the polar
coordinate pointing North, usually represented as upwards, and the polar angle is
measured counter-clockwise. The equivalent frame in python will be considering the
0 pointing downwards, and measuring the polar angle clockwise.

- Another convention to follow is that the orientation of the disk in the sky, i.e.,
the angle of the major axis Psi_0 is measured and reported on the red (receeding)
side of the galaxy. Therefore, after correcting Psi to represent (Psi - Psi_0), the
receeding side will be around Psi=0°, and the approaching side around Psi=180° .


Contributors
------------
  Juan Carlos Basto Pineda,
  Juan Manuel Pacheco,
  Fabricio Ferrari             


Updated
-------
June 3, 2021

"""


import numpy as np
import pandas as pd


class VelocityMaps(object):
    """
    Instance = VelocityMap(Vel_map,p_geo,beta=1,Err_map=1)
    
    This class will perform different actions related to 2D velocity maps.
    It provides methods to calculate intermediate geometrical quantities,
    and to compute the rotation curve by applying eliptical binning on the
    2D velocity map.

    Parameters
    ----------   
    Vel_map : ndarray
        observed velocity map in [km s^-1]
        
    pgeo : list or 1D array
        List of five numbers defining the geometrical orientation of the disk
        - pgeo = [psi0,i0,x0,y0,Vsys]
        
        - psi0: position angle of the major axis [deg]
        - i0: inclination of the disk in the sky [deg]
        - x0: row coordinate of the center of rotation [pix]
        - y0: column coordinate of the center of rotation [pix]
        - Vsys: Systemic velocity of the galaxy [km s^-1]

    beta : float
        pixel plate scale in units of [deg/pix] or [arcsec/pix] or [kpc/pix]
        It is set to 1 by default, giving the radii in pixels in that case
        
    Err_map : float or ndarray
        Measuremente errors in the projected velocities [km s^-1]
        If float, the same velocity error will be used in all measurements
        
    Notes
    -----
        In some cases it is necessary to add 90 deg to 'psi0' to set the right 
        frame of coordinates, it depends on the definition os the value of Psi0
        reported for the galaxy.
    """

    def __init__(self,Vel_map,p_geo,beta=1,Err_map=1.):
        """
        Initializing instance attributes and setting useful quantities up.
        
        
        Parameters
        ----------
        The necessary parameters to instatiate the class are described in the
        docstring of the definition of the class itself (see above).
        
        Returns
        -------
        self.Vp : observed velocity map in [km s^-1]
        self.p_geo : listd or ndarray [psi0,i0,x0,y0,Vsys] given as input
        self.beta : pixel scale in [deg/pix] or [arcsec/pix] or [kpc/pix]
        
        self.psi0 : position angle of the major axis [rad] (reeceding side)
        self.i0 : inclination of the disk in the sky [rad]
        self.x0 : row coordinate of the center of rotation [pix]
        self.y0 : column coordinate of the center of rotation [pix]
        self.Vsys : Systemic velocity of the galaxy [km s^-1]
        
        self.sin_i : sinus of inclination
        self.cos_i : cosinus of inclination
        
        self.dim : tuple (M,N), chape of the velocity map
        self.N : number of valid pixels in the velocity map
        self.index : positions of valid pixels in the velocity map
        self.index_nan : positions the NaNs in the velocity map
        
        self.sigmaVp : ndarray
            2D map of errors in the measured velocities
            Equal to Err_map if a ndarray is passed.
            If Err_map is a scalar, its valus is replicated in all pixels of self.sigmaVp              
        """
        
        self.Vp = Vel_map
        self.p_geo = p_geo
        self.beta = beta

        # Since now on self.psi0 and self.i0 will be in radians
        self.psi0 = np.radians(p_geo[0])
        self.i0 = np.radians(p_geo[1])
        self.x0 = p_geo[2]
        self.y0 = p_geo[3]
        self.Vsys = p_geo[4]

        # Compute trigonometric functions as they are constantly needed
        self.sin_i = np.sin(self.i0)
        self.cos_i = np.cos(self.i0)

        # Dimensions of the velocity map. Main arrays will have same dimensions
        self.dim = self.Vp.shape

        # Number of valid pixels, and positions of NANs and not NANs
        self.count_NaNs()
        
        # If the error is a float number, an error map is created replicating it 
        if isinstance(Err_map,np.ndarray):
            self.sigmaVp = Err_map
        else:
            self.sigmaVp = Err_map*np.ones(self.dim)
            self.sigmaVp[self.index_nan] = np.NaN
        
                
    def count_NaNs(self):
        """
        Count the number of valid pixels in the maps, and stores their positions
        as well as the positions of the NaNs. Can be run inmediately after self.__init__
        
        Parameters
        ----------
        
        Returns
        -------
        self.index : tuple
            position of valid pixels in the velocity map
        self.index_nan : tuple
            position of NaNs in the velocity map
        self.N : float
            Number of valid pixels in the velocity map
        """
        
        self.index = np.where(~np.isnan(self.Vp))
        self.index_nan = np.where(np.isnan(self.Vp))
        self.N = self.index[0].size
    

    def Geometry(self):
        """
        Given the geometrical parameters, compute the auxiliary matrices: r,alpha,psi.
        It also stores the trigonometric functions of psi as attributes of the object

        Parameters
        ----------
        Can be run inmediately after the __init__

        Returns
        -------
        self.r : ndnarray
            Matrix of true radii to each pixel in the plane of the disk
            Units will be given by the beta parameter
        self.alpha : ndarray
            Auxiliary matrix for distance transformations
        self.psi : ndarray
            Matrix of angular positions with respect to the major axis
        self.cos_psi : ndarray
            cosinus of psi at every pixel
        self.sin_psi : ndarray
            sinus of psi at every pixel
        """
        
        y,x = np.mgrid[:self.dim[0],:self.dim[1]]
        
        # with the following correction, self.psi represents (psi - psi_0)
        self.psi = np.arctan2(self.x0-x,y-self.y0)
        self.psi = np.where(self.psi>=self.psi0, self.psi-self.psi0, 2*np.pi + self.psi-self.psi0)
        # store the trigonometric values for quickness
        self.cos_psi = np.cos(self.psi)
        self.sin_psi = np.sin(self.psi)
        
        self.alpha = np.sqrt((self.cos_psi)**2 + (self.sin_psi)**2/self.cos_i**2)
        
        # radius to each pixel in the plane of the sky
        self.R = np.sqrt((x-self.x0)**2. + (y-self.y0)**2.)
        # real radii of each pixel to the center in the plane of the disk
        self.r = self.alpha*self.R
        # convert to alternative units if specified through the conversion factor beta
        self.r = self.r*self.beta
        
        
    def mask_angle(self,angle):
        """
        Excludes a given angular region around the minor axis, both in the velocity map
        and in the error map. Need to run self.Geometry() first.

        Parameters
        ----------
        angle : float
            angular width to be ignored on each side of minor axis [deg]

        Returns
        -------
        Updates the following attribures:
        - self.Vp
        - self.sigmaVp
        - self.N
        - self.index
        - self.index_nan
        """
        
        angle = np.radians(angle)
        
        # exclude angular regions around the 'positive' and the 'negative' minor axis
        to_reject_1 = (np.pi/2-angle <= self.psi) & (self.psi <= np.pi/2+angle)
        to_reject_2 = (3*np.pi/2-angle <= self.psi) & (self.psi <= 3*np.pi/2+angle)
        to_reject = to_reject_1 | to_reject_2
        
        self.Vp[to_reject] = np.NaN
        self.sigmaVp[to_reject] = np.NaN
        self.count_NaNs()
        
        self.get_index_sides(angle)
        
                        
        
    def get_index_sides(self,angle):
        """
        Get the index of the approaching and the receeding sides independently,
        excluding an angular region around the minor axis, given by 'angle'
        
        Parameters
        ----------
        angle : float
            Angular width to be ignored on each side of minor axis [rad]
        
        Returns
        -------
        self.index_blue : tuple
            position of valid pixels in the approaching side of the velocity map
        self.index_red : tuple
            position of valid pixels in the receeding side of the velocity map
        """
        
        index_blue = (np.pi/2+angle < self.psi) & (self.psi < 3*np.pi/2-angle)
        index_blue = index_blue & (~np.isnan(self.Vp))
        self.index_blue = np.where(index_blue)
        
        index_red = (np.pi/2+angle > self.psi) | (3*np.pi/2+angle < self.psi)
        index_red = index_red & (~np.isnan(self.Vp))
        self.index_red = np.where(index_red)


    def get_index_slit(self,slit_width):
        """
        Get the index of the positions along a central vertical stripe imitating
        the position of the long-slit
        
        Parameters
        ----------
        ncol : int
            Width of the slit in the units given by self.beta
            
        Returns
        -------
        self.index_ls : tuple
            

        """
        
        y_distance = np.abs(self.R * self.beta * self.sin_psi)
        mask_ls = (y_distance <= slit_width/2)
        self.index_ls = mask_ls & (~np.isnan(self.Vp))
        
        
    def deproject(self):
        """
        Calculates the circular velocity map by deprojecting the observed one
        self.Geometry() must be run in advance. An angular section around the
        minor axis might be filtered out first using self.mask_angle(), but this
        is optional
        
        Parameters
        ----------
        
        Returns
        -------
        self.Vc : ndarray
            Circular velocity associated to every pixel
        self.sigmaVc : ndarray
            Errors in the circular velocities
        """
        
        factor = self.alpha / (self.sin_i * self.cos_psi)
        self.Vc = (self.Vp - self.Vsys) * factor
        self.sigmaVc = self.sigmaVp * factor
        

    def cloudy_rotation_curve(self):
        """
        Creates a rotation curve by de-projecting all individual velocity pixels.
        It creates 3 versions of the rotation curve, one using all the data points,
        one using only the receeding part of the map, and one witht the approaching
        part. An angular section around the minor axis might be filtered out first
        using self.mask_angle(), but this is optional.
        
        Parameters
        ----------
        
        Returns
        -------
        sel.RC_all : ndarray
            - (N,3) being N the number of valid pixels
            - columns are [radius, Vcir, Error_Vcir], ordered by increasing radius
        self.RC_appr : ndarray
            Same as the former one, but for the approaching side only
        self.RC_receed : ndarray
            Same as the former ones, but for the receeding side only
        """
        
        self.RC_all = self.sort_curve(self.r,self.Vc,self.sigmaVc,self.index)
        
        # If angular filtering was not done, approaching/receeding indexes are missing
        if ~hasattr(self, 'index_red'):
            self.get_index_sides(angle=0)
            
        self.RC_appr = self.sort_curve(self.r,self.Vc,self.sigmaVc,self.index_blue)
        self.RC_receed = self.sort_curve(self.r,self.Vc,self.sigmaVc,self.index_red)
    
    
    def sort_curve(self,r_map,Vel_map,Err_map,index):
        """
        This module sort a lot of velocity points according to their radius.
        Inputs should be same-size matrices for radius, Velocity and Errors.
        An index tuple is also required. If there is no error map to be sorted,
        set that input to None.
        
        Parameters
        ----------
        r_map : ndarray
            Array of radii to the center of each velocity point
        Vel_map : ndarray
            Array if circular velocities
        Err_map : ndarray
            Array of errors in the circular velocities
            If None, it means that only radii and velocities are to be used
        index : tuple
            Positions of the arrays that are to be taken into account
            If None, all pixels are to be considered
            
        Returns
        -------
        srt_curve : ndarray
            Sorted rotation curve
            - (N,3) columns are [radius,Vc,ErrVc]
            - (N,2) columns are [radius,Vc]            
        """
        
        if index is None:
            index = np.ones(r_map.shape).astype(bool)
        
        # index of the sorted radii
        hh = np.argsort(np.ravel(r_map[index]))
        
        if type(Err_map) != None:
            srt_curve = np.zeros([r_map[index].size,3])
            srt_curve[:,2] = np.ravel(Err_map[index])[hh[:]]
        else:
            srt_curve = np.zeros([r_map[index].size,2])
        srt_curve[:,0] = np.ravel(r_map[index])[hh[:]]
        srt_curve[:,1] = np.ravel(Vel_map[index])[hh[:]]
                                                  
        return srt_curve


    def bin_one_curve(self,RC,binsize):
        """
        Binning of the rotation curve from deprojection of all velocity pixels.
        
        Parameters
        ----------
        RC : ndarray
            Rotation curve in a 2-column or 3-column array
            Columns are [radius,vel,err_vel]
        binsize : float
            Width of the radial bins
        
        Returns
        -------
        RC_binned : ndarray
            Binned rotation curve with the same structure of the input
        """
        bins = np.arange(0,np.max(RC[:,0])+binsize,binsize)
        belong_bins = np.digitize(RC[:,0],bins)
        df = pd.DataFrame({'bin':belong_bins, 'radius':RC[:,0],  'vel':RC[:,1]})
        vel_mean = df.groupby('bin').apply(lambda x: np.mean(x.vel))
        vel_std = df.groupby('bin').apply(lambda x: np.std(x.vel))
        r_mean = df.groupby('bin').apply(lambda x: np.mean(x.radius))
        RC_bin = np.array(pd.concat([r_mean, vel_mean, vel_std],axis=1))
        return RC_bin
    
    
    def bin_all_curves(self,binsize):
        """
        Bin the three rotations curves obtained by deprojection
        
        Parameters
        ----------
        binsize : float
            Width of the radial bins
            
        Returns
        -------
        self.RC_all_bin : ndarray
            Binned rotation curve, from all pixels in the velocity map
        self.RC_appr_bin : ndarray
            Binned rotation curve, from the approaching side of the velocity map
        self.RC_receed_bin : ndarray
            Binned rotation curve, from the receeding side of the velocity map
        
        """
        self.RC_all_bin = self.bin_one_curve(self.RC_all,binsize)
        self.RC_appr_bin = self.bin_one_curve(self.RC_appr,binsize)
        self.RC_receed_bin = self.bin_one_curve(self.RC_receed,binsize)

        
        


