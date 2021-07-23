## This is the first version of velocity maps
from typing import Sequence
import numpy as np
from numpy import r_


class Galaxie:

    def __init__(self, vel_map, p_geo, beta=1,Err_map=1):

        # Velocity map
        self.vp = vel_map

        # List with the geometrical parameters
        self.p_geo = p_geo

        # Conversion factor from
        self.beta = beta

        # Extract the geometrical parameters individually
        # Since now on self.psi0 and self.io will be in radians
        self.psi0 = np.radians(p_geo[0])
        self.i0 = np.radians(p_geo[1])
        self.x0 = p_geo[2]
        self.y0 = p_geo[3]
        self.vsys = p_geo[4]

        # Compute trigonometric functions
        self.sin_i = np.sin(self.i0)
        self.cos_i = np.cos(self.i0)

        # Dimensions of the velocity map. Main arrays will have same dimensions
        self.dim = self.vp.shape

        #Number of valid pixels, and where Nan's are
        self.index = np.where(~np.isnan(self.vp))
        self.index_nan = np.where(np.isnan(self.vp))
        # Number of not 'NaN' values.
        self.N = self.index[0].size

        # If the error is a float number, an error map is created replicating it
        if isinstance(Err_map, float):
            self.sigmaVp = Err_map*np.ones(self.dim)
            self.sigmaVp[self.index_nan] = np.NaN
        else:
            self.sigmaVp = Err_map
        

        # Create the auxiliry matrices self.r, self.alpha, self.psi
        self.Geometry()

    def Geometry(self):


        y,x = np.mgrid[:self.dim[0],:self.dim[1]]
        R = np.sqrt((x-self.x0)**2. + (y-self.y0)**2.)
        self.psi = np.arctan2(y-self.y0,self.x0-x) + np.pi
        self.psi = np.where(self.psi>=self.psi0, self.psi-self.psi0, 2*np.pi + self.psi-self.psi0)
        self.cos_psi = np.cos(self.psi)
        self.sin_psi = np.sin(self.psi)
        self.alpha = np.sqrt((self.cos_psi)**2 + (self.sin_i)**2/self.cos_i**2)
        self.r = self.alpha*R
        # Convert to angular units if specified
        self.r = self.r*self.beta

    
    # This modul calculate the 
    def velocity(self, v_t):
        v_cir = -v_t*np.e**(-self.r/self.r_t)+v_t  
        v_cir[v_cir > -v_t*np.e**(-self.r_max/self.r_t)+v_t ] = 'nan'
        return v_cir


    def deproject(self):
        factor = self.alpha / (self.sin_i * self.cos_psi)
        self.vc = (self.vp-self.vsys) * factor
        self.sigma_Vc = self.sigmaVp * factor
    

        # Calculates the velocity projected along the line-of-sight.
    def project(self):
        self.v_los = self.vsys + (self.vc/self.alpha)*np.sin(self.i0)*np.cos((self.psi-self.psi0))
        

    

    


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














