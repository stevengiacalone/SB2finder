import numpy as np
import pandas as pd
import scipy.interpolate as inter 
from astropy.io import fits
from astropy.table import Table
from scipy.constants import c
from numpy import interp

def spline_fit(x,y,window):
    breakpoint = np.linspace(np.min(x),np.max(x),int((np.max(x)-np.min(x))/window))
    ss = inter.LSQUnivariateSpline(x,y,breakpoint[1:-1])
    return ss

def flatspec_spline(x,rawspec,weight, order=8, n_iter = 5, ffrac = 0.985):
    
    #If array is decreasing, we apply a flip to make it increasing
    if x[0] > x[1]:
        x = np.flip(x)
        rawspec = np.flip(rawspec)
        weight = np.flip(weight)

    pos = np.where((np.isnan(rawspec)==False) & (np.isnan(weight)==False))[0]

    ss = spline_fit(x[pos],rawspec[pos],5.)#5 Ang as knot points for flattening continuum
    yfit = ss(x)

    for i in range(n_iter):
        normspec = rawspec / yfit
        pos = np.where((normspec >= ffrac) & (yfit > 0))[0]#& (normspec <= 2.)

        ss = spline_fit(x[pos],rawspec[pos],5.)
        yfit = ss(x)
    normspec = rawspec / yfit

    return x,rawspec,weight,normspec,yfit


def get_spectrum(file_name, trace_num, color):
    """
    Retrieves spectrum with wavelength solutions from L1 files.
    Arguments:
        file_name: Name of L1 file with spectrum (string)
        trace_num: Trace number of the L1 file (either 1,2, or 3) (int or float)
        color: Either 'GREEN' or 'RED' (string)
    Returns:
        SCI_WAVE: Wavelength of spectrum (Angstroms)
        SCI_FLUX: Flux density of spectrum
    """
    
    L1_file = file_name
    L1 = fits.open(file_name)
    
    SCI_WAVE = np.array(L1[color + '_SCI_WAVE' + str(trace_num)].data)
    SCI_FLUX = np.array(L1[color + '_SCI_FLUX' + str(trace_num)].data)
            
    return SCI_WAVE, SCI_FLUX


def combine_spectra(synth_wave, synth_flux, wave, flux, rv, a, vsini):
    #wavelengths and fluxes have been flatten using flatspec_spline
    #wave and flux are single orders of the spectra
    #'a' is our brightness scale factor of the companion star
    
    synth_flux_broad = broaden_spec(synth_wave, synth_flux, vsini)
    
    new_wave_synth = synth_wave * (1 + (rv/(c/1000)))

    flux_interpolated = np.interp(wave, new_wave_synth, synth_flux_broad)
    combine_flux = (a*flux_interpolated+flux) / (1 + a)
    
    return wave, combine_flux


def broaden_spec(w, s, vsini, eps=0.6, nr=10, ntheta=100, dif = 0.0):
    '''
    A routine to quickly rotationally broaden a spectrum in linear time.
    Function from https://github.com/Adolfo1519/RotBroadInt/tree/main

    INPUTS:
    w - wavelength scale of the input spectrum
    
    s - input spectrum
    
    vsini (km/s) - projected rotational velocity
    
    OUTPUT:
    ns - a rotationally broadened spectrum on the wavelength scale w

    OPTIONAL INPUTS:
    eps (default = 0.6) - the coefficient of the limb darkening law
    
    nr (default = 10) - the number of radial bins on the projected disk
    
    ntheta (default = 100) - the number of azimuthal bins in the largest radial annulus
                            note: the number of bins at each r is int(r*ntheta) where r < 1
    
    dif (default = 0) - the differential rotation coefficient, applied according to the law
    Omeg(th)/Omeg(eq) = (1 - dif/2 - (dif/2) cos(2 th)). Dif = .675 nicely reproduces the law 
    proposed by Smith, 1994, A&A, Vol. 287, p. 523-534, to unify WTTS and CTTS. Dif = .23 is 
    similar to observed solar differential rotation. Note: the th in the above expression is 
    the stellar co-latitude, not the same as the integration variable used below. This is a 
    disk integration routine.

    '''

    ns = np.copy(s)*0.0
    tarea = 0.0
    dr = 1./nr
    for j in range(0, nr):
        r = dr/2.0 + j*dr
        area = ((r + dr/2.0)**2 - (r - dr/2.0)**2)/int(ntheta*r) * (1.0 - eps + eps*np.cos(np.arcsin(r)))
        for k in range(0,int(ntheta*r)):
            th = np.pi/int(ntheta*r) + k * 2.0*np.pi/int(ntheta*r)
            if dif != 0:
                vl = vsini * r * np.sin(th) * (1.0 - dif/2.0 - dif/2.0*np.cos(2.0*np.arccos(r*np.cos(th))))
                ns += area * np.interp(w + w*vl/2.9979e5, w, s)
                tarea += area
            else:
                vl = r * vsini * np.sin(th)
                ns += area * np.interp(w + w*vl/2.9979e5, w, s)
                tarea += area
          
    return ns/tarea