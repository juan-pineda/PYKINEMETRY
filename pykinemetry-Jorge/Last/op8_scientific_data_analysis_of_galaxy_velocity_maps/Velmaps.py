#!/usr/bin/env python
"""
Scientific Data Analysis of Galaxy Velocity Maps

This code constructs a galaxy model from its experimental velocity map.

Authors:
        José Miguel Ladino (CORREO), Universidad Nacional de Colombia
        Omar Asto Rojas (CORREO), Uiversidad Nacional de Ingenería, Perú
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
from mpl_toolkits.axes_grid1 import make_axes_locatable


### Part 1: Initial modeling, synthetic data and error estimation
def Modelo_Vlos(xx , yy, p):
    """    
        Compute the Velocity map in the line of observation of a galaxy data.
        Returns a 2D array with the velocity values.
        
        Parameters
        ----------
        xx : array_like
             array contain the x grid of a 2D map grid.
        yy : array_like
             array contain the y grid of a 2D map grid.
        p : array_like
            Physical parameters the observated galaxy
            example: 
                p = [x0, y0, v_sys, i, phi_0, V_t, R_0, a, g]
                where,
                    x0 : x coordinate of the galaxy center
                    y0 : y coordinate of the galaxy center
                    v_sys : velocity of the galaxy center
                    i : zenith angle of inclination in the galaxy
                    phi_0 : Angle for the projected disk
                    V_t : Transversal velocity
                    R_0 : Observed radius
                    a : Adimentonal fit parameter
                    g : Adimentional fit parameter
        Returns
        -------      
        V_los : Velocity in observational line
---------------------------------------------------------------------
             
    """
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
    alpha = alpha2(phi, p[4], p[3])
  
    r=alpha*R

    v_c = p[5] * ((r/p[6])**(p[7]))/(1 + (r/p[6])**(p[8]))
    V_los = p[2] + v_c*((np.sin(np.deg2rad(p[3])))*np.cos(np.deg2rad(phi - p[4])))/alpha

    return V_los

######################### ERROR Y MASK #############################
def error(tpl, x, y, z):
    """
            Compute the error between model and data array.
            Returns a 2D array with the error values.

            Parameters
            ----------
            x : array_like
                 array contain the x grid of a 2D map grid.
            y : array_like
                 array contain the y grid of a 2D map grid.
            tpl : array_like
                Physical parameters the observated galaxy
                example: 
                    p = [x0, y0, v_sys, i, phi_0, V_t, R_0, a, g]
                    where,
                        x0 : x coordinate of the galaxy center
                        y0 : y coordinate of the galaxy center
                        v_sys : velocity of the galaxy center
                        i : zenith angle of inclination in the galaxy
                        phi_0 : Angle for the projected disk
                        V_t : Transversal velocity
                        R_0 : Observed radius
                        a : Adimentonal fit parameter
                        g : Adimentional fit parameter
             z : array_like
                 2D data grid map.
            Returns
            -------      
            Array : np.ravel(E[~mask])
---------------------------------------------------------------------
           
    """
    mask = np.isnan(z)   
    E =  Modelo_Vlos(x, y, tpl) - z
    
    return np.ravel(E[~mask])

######################### FIT A PARTIR DE DATOS ####################
def fit_vel(file_dir,typef_bool,X,Y,p,boolean):
    """
    
            This function implements the least squares method, to minimise the discrepancies
            between the velocity values of each pixel between the real or synthetic velocity
            map and the model velocity map. The function to be used is \textit{leastsq} of 
            Scipy, which will be applied to the function -error- defined between the difference
            of velocity values of the real or synthetic data and the model, to determine the 
            values of the 9 model parameters that best fit the real or synthetic velocity map.

    
            How to use me?: 
                        1) Type your y and x vectors and you matrix before:
                            x = np.arange(0,160)
                            y = np.arange(0,160)
                            Y, X = np.meshgrid(x,y)
                        2) Define the values of p = [x0, y0, v_sys, i, phi_0, V_t, R_0, a, g]
                        where,
                            x0 : x coordinate of the galaxy center
                            y0 : y coordinate of the galaxy center
                            v_sys : velocity of the galaxy center
                            i : zenith angle of inclination in the galaxy
                            phi_0 : Angle for the projected disk
                            V_t : Transversal velocity
                            R_0 : Observed radius
                            a : Adimentonal fit parameter
                            g : Adimentional fit parameter

                        3) Write: fit_vel(file_dir,typef_bool,x,y,p,boolean) 
                        where boolean are:
                        'True' or 'False' words and typef especify the type of
                        file (False if is .txt, True if is .fits)

                        4) If your boolean keyword is 'True', fit_vel shows you the histogram of 
                        the residual values obtained between data and fit model.
            Parameters
            ----------
            file_dir : string
                 string with the name of file (txt or fits) type in quotes 
            typef_bool : boolean
                 Type True if your file is -fits- and False if is -txt-
            x : array_like
                 array contain the x grid of a 2D map grid.
            y : array_like
                 array contain the y grid of a 2D map grid.
            p : List of integers
                p = [x0, y0, v_sys, i, phi_0, V_t, R_0, a, g]
                        where,
                            x0 : x coordinate of the galaxy center
                            y0 : y coordinate of the galaxy center
                            v_sys : velocity of the galaxy center
                            i : zenith angle of inclination in the galaxy
                            phi_0 : Angle for the projected disk
                            V_t : Transversal velocity
                            R_0 : Observed radius
                            a : Adimentonal fit parameter
                            g : Adimentional fit parameter
            
            Returns
            -------      
            F : 2D array fitted model
            Plot: Data + Model + Residual
            Optional plot if True: Residual distribution
---------------------------------------------------------------------
            
    """
    if typef_bool == False:
        Z = np.loadtxt(file_dir)
        x = np.arange(0,len(Z))
        y = np.arange(0,len(Z))
        Y, X = np.meshgrid(x,y)
    elif typef_bool == True:
        hdu = fits.open(file_dir)
        Z= hdu[0].data
        x = np.arange(0,len(Z))
        y = np.arange(0,len(Z))
        Y, X = np.meshgrid(x,y)
    #x = np.arange(0,160)
    #y = np.arange(0,160)
    #Y, X = np.meshgrid(x,y)
    best = leastsq(error, p, args= (X, Y, Z), full_output=1)
    F = Modelo_Vlos(X, Y, best[0])
    
    #PLOT
      #  if typef_bool == False:
    fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(9, 6))
    fig.subplots_adjust(hspace=2.5, left=0.5, right=1.5)

    ax = axs[0]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    hb = ax.imshow(F+Z-Z,cmap="gnuplot2")
    ax.set_title("Model")
    cb = fig.colorbar(hb, ax=ax, cax=cax)


    ax = axs[1]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    hb = ax.imshow(Z,cmap="gnuplot2")
    ax.set_title("Data")
    cb = fig.colorbar(hb, ax=ax, cax=cax)


    ax = axs[2]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    hb = ax.imshow(F-Z,cmap="gnuplot2")
    ax.set_title("Resta")
    cb = fig.colorbar(hb, ax=ax, cax=cax)


    plt.show()
           
    
######################### HISTOGRAMA RESIDUO ####################
    if boolean==True:
        
        sb.displot((F-Z).ravel(), color='#F2AB6D', bins=10, kde=False)
        plt.xlabel('Residuo')
        plt.ylabel('Frecuencia')
        #plt.hist((F-Z).ravel(), bins = 15)
        plt.show()
    return F

####### Fit de parámetros a partir de las covarianzas ##########

def fit_cov(file_dir,typef_bool, X, Y, variance, n):
    """    
            This function performs an optimisation of the parameter values that can
            be found by the \textit{leastsq} function, making use of the covariance
            matrix $Cov$ with which an estimate of the variance $Var$ can be calculated
            for each parameter among the data analysed in the error function, as follows:
            
                     Var = Tr(Cov) (\frac{z_{fit}-z}{n-p})
                     
            where $Tr(Cov)$ is the sum of the elements of the diagonal of the covariance
            matrix of each parameter, $z_{fit}$ is the velocity values using the fitted
            parameters, $z$ is the velocity values of the real or synthetic data set
            to be analysed, $n$ is the number of data in $z$ and $p$ is the number of
            parameters to be fitted.  
    
            Parameters
            ----------
            file_dir : string
                 string with the name of file (txt or fits) type in quotes 
            typef_bool : boolean
                 Type True if your file is -fits- and False if is -txt-
            x : array_like
                 array contain the x grid of a 2D map grid.
            y : array_like
                 array contain the y grid of a 2D map grid.
            variance : integer
                integer with a variance value
            n : integer
                times for compute the covariance
            
            Returns
            -------      
            bestparam : List of a params p for make the fit.
---------------------------------------------------------------------
 
    """    
    
    p = [80,80,5,30,-90,350,10,0.9,1.1] 
    ##Define parameters
    if typef_bool == False:
        Z = np.loadtxt(file_dir)
        x = np.arange(0,len(Z))
        y = np.arange(0,len(Z))
        Y, X = np.meshgrid(x,y)
    elif typef_bool == True:
        hdu = fits.open(file_dir)
        Z= hdu[0].data
        x = np.arange(0,len(Z))
        y = np.arange(0,len(Z))
        Y, X = np.meshgrid(x,y)
    sigma = np.sqrt(variance)
    ########
    Values = []
    P = []  
    for i in range(len(p)):
        x = np.linspace(p - 3*sigma, p + 3*sigma, n)
        P.append(x)
    P = np.array(P)
    P = P[0,:,:]
    best1  = []
    SumVar =[]
    bestVar=[]
    #print(P)
    for i in range(9):
        for j in range(n):
            best = leastsq(error, P[j,], args= (X, Y, Z), full_output=1)
            best1.append(best[0])
            Mcov=best[1]
         #   print(np.shape(Mcov))
         #   print(Mcov)
            Diag=[]
            for k in range(9):
                if Mcov is not None : 
                    Diag.append(Mcov[k][k])
            R1=np.nan_to_num(Modelo_Vlos(X, Y, best[0]))
            R2=np.nan_to_num(Z)
            Res= (R1-R2)**2 / (len(Z)-9)**2
            Res1=sum( Res)
            Var=[]
            for l in Diag:
                Incer=l*sum(Res1)
                Var.append(Incer)  
            bestVar.append(Var)
            SumVar.append(sum(np.absolute(Var)))
        Var = np.array(Var)
        SumVar = np.array(SumVar)
        minim =np.min(SumVar[np.nonzero(SumVar)])
        indexm = int(np.array(np.where(SumVar == minim)))
        bestparam=[best1[indexm], np.array(bestVar[indexm])]
        return bestparam
    
##############Fit con varias tuplas de parámetros ##############

def fit_tuplas(file_dir,typef_bool, X, Y, variance, n, boolean):
    """    
            This function compute que velocity map from the bestparams in fit_cov function
    
            Parameters
            ----------
            file_dir : string
                 string with the name of file (txt or fits) type in quotes 
            typef_bool : boolean
                 Type True if your file is -fits- and False if is -txt-
            x : array_like
                 array contain the x grid of a 2D map grid.
            y : array_like
                 array contain the y grid of a 2D map grid.
            variance : integer
                integer with a variance value
            n : integer
                times for compute the covariance
            boolean : boolean
                 True for plot the residual distribution (histogram)

            Returns
            -------      
            F_1 : Fitted velocity map
            Plot: Data + Model + Residual
            Optional plot if True: Residual distribution 
---------------------------------------------------------------------

    """    
    if typef_bool == False:
        Z = np.loadtxt(file_dir)
        x = np.arange(0,len(Z))
        y = np.arange(0,len(Z))
        Y, X = np.meshgrid(x,y)
    elif typef_bool == True:
        hdu = fits.open(file_dir)
        Z= hdu[0].data
        x = np.arange(0,len(Z))
        y = np.arange(0,len(Z))
        Y, X = np.meshgrid(x,y)
        
    bes_er = fit_cov(file_dir,typef_bool, X, Y, variance, n)
    F_1 = Modelo_Vlos(X, Y, bes_er[0])
    
############################# PLOTTING #########################
    fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(9, 6))
    fig.subplots_adjust(hspace=2.5, left=0.5, right=1.5)

    ax = axs[0]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    hb = ax.imshow(F_1-Z+Z,cmap="gnuplot2")
    ax.set_title("Model")
    cb = fig.colorbar(hb, ax=ax , cax=cax)

    ax = axs[1]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    hb = ax.imshow(Z,cmap="gnuplot2")
    ax.set_title("Data")
    cb = fig.colorbar(hb, ax=ax , cax=cax)


    ax = axs[2]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    hb = ax.imshow(F_1-Z,cmap="gnuplot2")
    ax.set_title("Resta")
    cb = fig.colorbar(hb, ax=ax , cax=cax)

    plt.show()

    if boolean==True:
        sb.displot((F_1-Z).ravel(), color='#F2AB6D', bins=20)
        plt.xlabel('Residuo')
        plt.ylabel('Frecuencia')
        plt.show()
        print(np.nansum(np.absolute((F_1-Z).ravel())))
        
    return F_1













