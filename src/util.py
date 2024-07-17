import numpy as np
import pandas as pd
import scipy.interpolate as inter 
from astropy.io import fits
from astropy.table import Table

c = 3e5 # speed of light in km/s

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

    return normspec,yfit


def get_spectrum(file_name, trace_num, color):
    L1_file = file_name
    L1 = fits.open(file_name)
    
    if color == 'GREEN':
        if trace_num == 1:
            SCI_WAVE = np.array(L1['GREEN_SCI_WAVE1'].data)
            SCI_FLUX = np.array(L1['GREEN_SCI_FLUX1'].data)
        elif trace_num == 2:
            SCI_WAVE = np.array(L1['GREEN_SCI_WAVE2'].data)
            SCI_FLUX = np.array(L1['GREEN_SCI_FLUX2'].data)
        else:
            SCI_WAVE = np.array(L1['GREEN_SCI_WAVE3'].data)
            SCI_FLUX = np.array(L1['GREEN_SCI_FLUX3'].data)    
    else:
        if trace_num == 1:
            SCI_WAVE = np.array(L1['RED_SCI_WAVE1'].data)
            SCI_FLUX = np.array(L1['RED_SCI_FLUX1'].data)
        elif trace_num == 2:
            SCI_WAVE = np.array(L1['RED_SCI_WAVE2'].data)
            SCI_FLUX = np.array(L1['RED_SCI_FLUX2'].data)
        else:
            SCI_WAVE = np.array(L1['RED_SCI_WAVE3'].data)
            SCI_FLUX = np.array(L1['RED_SCI_FLUX3'].data)
            
    return SCI_WAVE, SCI_FLUX


def combine_spectra(synth_wave, synth_flux, wave, flux, rv):
    #wavelengths and fluxes have been flatten using flatspec_spline
    #wave and flux are single orders of the spectra
    
    new_wave_synth = synth_wave * (1 + (rv/c))

    flux_interpolated = np.interp(wave, new_wave_synth, synth_flux)
    combine_flux = (flux_interpolated+flux) / 2
    
    return wave, combine_flux

