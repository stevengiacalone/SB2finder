import numpy as np
import pandas as pd
import scipy.interpolate as inter 
from astropy.io import fits
from astropy.table import Table
from scipy.constants import c
from numpy import interp

def synth_flux_correction(flux, Teff):
    """
    Applies correction factor to PHOENIX spectra flux so that it corresponds to the
    actual flux of the star. Correction factors scale as L/R^2 using the following table
    https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    Args:
        flux: Flux of synthetic spectrum (numpy array)
        Teff: Effective temperature of star (int or float).
    Returns:
        Flux multiplied by correction factor.
    """
    st_props = np.loadtxt("../../stellar_properties.txt", skiprows=1, usecols=[1,4,6]).T
    Teffs = st_props[0]
    Lums = 10**(st_props[1])
    Rads = st_props[2]
    corrections = Lums/Rads**2
    
    this_idx = np.argmin(np.abs(Teffs - Teff))
    this_correction = corrections[this_idx]
    return flux*this_correction

def spline_fit(x,y,window):
    breakpoint = np.linspace(np.min(x),np.max(x),int((np.max(x)-np.min(x))/window))
    ss = inter.LSQUnivariateSpline(x,y,breakpoint[1:-1])
    return ss

def flatspec_spline(x,rawspec,weight, order = 8, n_iter = 5, ffrac = 0.985, window = 5):
    
    #If array is decreasing, we apply a flip to make it increasing
    if x[0] > x[1]:
        x = np.flip(x)
        rawspec = np.flip(rawspec)
        weight = np.flip(weight)

    pos = np.where((np.isnan(rawspec)==False) & (np.isnan(weight)==False))[0]

    ss = spline_fit(x[pos],rawspec[pos],window)
    yfit = ss(x)

    for i in range(n_iter):
        normspec = rawspec / yfit
        pos = np.where((normspec >= ffrac) & (yfit > 0))[0]#& (normspec <= 2.)

        ss = spline_fit(x[pos],rawspec[pos],window)
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
        SCI_WAVE: Wavelength of spectrum [Angstroms] (numpy array)
        SCI_FLUX: Flux density of spectrum (numpy array)
    """
    
    L1_file = file_name
    L1 = fits.open(file_name)
    SCI_WAVE = np.array(L1[color + '_SCI_WAVE' + str(trace_num)].data)
    SCI_FLUX = np.array(L1[color + '_SCI_FLUX' + str(trace_num)].data)
    L1.close()
            
    return SCI_WAVE, SCI_FLUX


def combine_spectra(synth_wave, synth_flux, wave, flux, rv, a, vsini):
    #wavelengths and fluxes have been flatten using flatspec_spline
    #wave and flux are single orders of the spectra
    #'a' is our brightness scale factor of the companion star
    
    synth_flux_broad = broaden_spec(synth_wave, synth_flux, vsini)
    
    new_wave_synth = synth_wave * (1 + (rv/(c/1000)))

    flux_interpolated = np.interp(wave, new_wave_synth, synth_flux_broad)
    a_interpolated = np.interp(wave, new_wave_synth, a)
    combine_flux = (flux_interpolated + a_interpolated*flux) / (1 + a_interpolated)
    
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

def merge_spec(file_name, color):
    
    #Retrieve the spectra from each trace
    wave1, flux1 = get_spectrum(file_name, 1, color)
    wave2, flux2 = get_spectrum(file_name, 2, color)
    wave3, flux3 = get_spectrum(file_name, 3, color)
    
    #Combine three traces into one
    avg_SCI_WAVE = np.empty_like(wave1)
    sum_SCI_FLUX =  np.empty_like(flux1)

    if color == 'GREEN':
        range_num = 35
    elif color == 'RED':
        range_num = 32
            
    for o in range(range_num):
        avg_SCI_WAVE[o,:] = np.flip((wave1[o,:] + wave2[o,:] + wave3[o,:]) / 3)
        sum_SCI_FLUX[o,:] = np.flip(flux1[o,:] + flux2[o,:] + flux3[o,:])
        
    #set range to be one less than number of orders
    # Initialize an empty 2D array
    stitched_wave = []
    stitched_flux = []
    matching_waves = []

    wave_1D = np.array([])
    flux_1D = np.array([])
    
    if color == 'GREEN':

        for o in range(range_num - 1):

            #Set the left order and the right order
            wave_left = avg_SCI_WAVE[o, :]
            flux_left = sum_SCI_FLUX[o, :]
            wave_right = avg_SCI_WAVE[o+1, :]
            flux_right = sum_SCI_FLUX[o+1, :]

            weight_left = np.ones(len(wave_left))
            weight_right =  np.ones(len(wave_right))

            flat_wave_right,rawflux_right,_,normspec_right,yfit_right = flatspec_spline(wave_right,flux_right,weight_right)

            flat_wave_left,rawflux_left,_,normspec_left,yfit_left = flatspec_spline(wave_left,flux_left,weight_left)

            #Set a mask to avoid 2 solutions since the yfit is parabolic
            max_right = np.argmax(yfit_right)
            max_left = np.argmax(yfit_left)
            yfit_right_masked = yfit_right[: max_right]
            yfit_left_masked = yfit_left[max_left :]

            for i in range(0, len(yfit_left_masked)):
                if o < 29:
                    matching_idx = np.argmin(np.abs(yfit_left_masked[i] - yfit_right_masked[:]))

                    if (flat_wave_right[: max_right][matching_idx-1] < flat_wave_left[max_left :][i]) & (flat_wave_right[: max_right][matching_idx+1] > flat_wave_left[max_left :][i]):
                        matching_wave = flat_wave_right[: max_right][matching_idx]
                        break
                else:
                    if yfit_left_masked[i] < yfit_right_masked[0]:
                        matching_wave =  flat_wave_left[max_left :][i]
                        break

            matching_waves.append(matching_wave)

            if o == 0:
                trim_mask = flat_wave_left < matching_wave
            else:
                trim_mask = (flat_wave_left < matching_wave) & (flat_wave_left > matching_waves[o-1])

            stitched_wave.append(np.array(flat_wave_left[trim_mask]))
            stitched_flux.append(np.array(rawflux_left[trim_mask]))

            wave_1D = np.concatenate([wave_1D, flat_wave_left[trim_mask]])
            flux_1D = np.concatenate([flux_1D, normspec_left[trim_mask]])
            
            if o == (range_num - 2):
                stitched_wave.append(np.array(flat_wave_right))
                stitched_flux.append(np.array(rawflux_right))

                wave_1D = np.concatenate([wave_1D, flat_wave_right])
                flux_1D = np.concatenate([flux_1D, normspec_right])

            
    elif color == 'RED':
        for o in range(range_num - 1):

            #Set the left order and the right order
            wave_left = avg_SCI_WAVE[o, :]
            flux_left = sum_SCI_FLUX[o, :]
            wave_right = avg_SCI_WAVE[o+1, :]
            flux_right = sum_SCI_FLUX[o+1, :]

            weight_left = np.ones(len(wave_left))
            weight_right =  np.ones(len(wave_right))

            flat_wave_right,rawflux_right,_,normspec_right,yfit_right = flatspec_spline(wave_right,flux_right,weight_right)

            flat_wave_left,rawflux_left,_,normspec_left,yfit_left = flatspec_spline(wave_left,flux_left,weight_left)

            #Set a mask to avoid 2 solutions since the yfit is parabolic
            max_right = np.argmax(yfit_right)
            max_left = np.argmax(yfit_left)
            yfit_right_masked = yfit_right[: max_right]
            yfit_left_masked = yfit_left[max_left :]

            for i in range(0, len(yfit_left_masked)):

                if flat_wave_left[max_left :][i] > flat_wave_right[: max_right][0]:
                    matching_wave =  flat_wave_left[max_left :][i]
                    break

            matching_waves.append(matching_wave)

            if o == 0:
                trim_mask = flat_wave_left < matching_wave
            else:
                trim_mask = (flat_wave_left < matching_wave) & (flat_wave_left > matching_waves[o-1])
            
            if o < 20:
                stitched_wave.append(np.array(flat_wave_left[trim_mask]))
                stitched_flux.append(np.array(rawflux_left[trim_mask]))

                wave_1D = np.concatenate([wave_1D, flat_wave_left[trim_mask]])
                flux_1D = np.concatenate([flux_1D, normspec_left[trim_mask]])
            else:
                stitched_wave.append(np.array(flat_wave_left))
                stitched_flux.append(np.array(rawflux_left))

                wave_1D = np.concatenate([wave_1D, flat_wave_left])
                flux_1D = np.concatenate([flux_1D, normspec_left])

            if o == (range_num - 2):
                stitched_wave.append(np.array(flat_wave_right))
                stitched_flux.append(np.array(rawflux_right))

                wave_1D = np.concatenate([wave_1D, flat_wave_right])
                flux_1D = np.concatenate([flux_1D, normspec_right])
        
    return stitched_wave, stitched_flux, wave_1D, flux_1D

def stitch_spec(file_name):
    """
    Stitches green and red CCD parts of the spectrum from L1 files.
    Arguments:
        file_name: Name of L1 file with spectrum (string)
    Returns:
        full_spectra_wave: Wavelength of entire spectrum [Angstroms] (list, each order is a numpy array)
        full_spectra_flux: Flux density of entire spectrum (list, each order is a numpy array)
        full_flat_wave: Flattened wavelength of entire spectrum [Angstroms] (numpy array)
        full_spectra_flux: Flattened flux density of entire spectrum (numpy array)
    """
    
    stitched_wave_green, stitched_flux_green, wave_1D_green, flux_1D_green = merge_spec(file_name, 'GREEN')
    stitched_wave_red, stitched_flux_red, wave_1D_red, flux_1D_red = merge_spec(file_name, 'RED')
    
    full_spectra_wave = stitched_wave_green + stitched_wave_red
    full_spectra_flux = stitched_flux_green + stitched_flux_red
    
    max_right = np.argmax(full_spectra_wave[35])
    max_left = np.argmax(full_spectra_wave[34])
    
    weight_left = np.ones(len(full_spectra_wave[34]))
    weight_right =  np.ones(len(full_spectra_wave[35]))

    flat_wave_right,rawflux_right,_,normspec_right,yfit_right = flatspec_spline(full_spectra_wave[35],full_spectra_flux[35],weight_right)

    flat_wave_left,rawflux_left,_,normspec_left,yfit_left = flatspec_spline(full_spectra_wave[34],full_spectra_flux[34],weight_left)

    #Set a mask to avoid 2 solutions since the yfit is parabolic
    max_right = np.argmax(yfit_right)
    max_left = np.argmax(yfit_left)
    yfit_right_masked = yfit_right[: max_right]
    yfit_left_masked = yfit_left[max_left :]

    for i in range(0, len(yfit_right_masked)):

        if flat_wave_right[: max_right][i] > flat_wave_left[max_left :][-1]:
            matching_wave = flat_wave_right[: max_right][i]
            break

    trim_mask = (flat_wave_right > matching_wave)
    
    full_spectra_wave[35] = full_spectra_wave[35][trim_mask]
    full_spectra_flux[35] = full_spectra_flux[35][trim_mask]
    
    flux_1D_red = flux_1D_red[wave_1D_red > matching_wave]
    wave_1D_red = wave_1D_red[wave_1D_red > matching_wave]
    
    full_flat_wave = np.concatenate((wave_1D_green, wave_1D_red))
    full_flat_flux = np.concatenate((flux_1D_green, flux_1D_red))
    
    telluric_mask = ~( 
    ((full_flat_wave > 7590) & (full_flat_wave < 7710)) |
    ((full_flat_wave > 6860) & (full_flat_wave < 7080)) | 
    ((full_flat_wave > 6270) & (full_flat_wave < 6330)) | 
    ((full_flat_wave > 5870) & (full_flat_wave < 6000)) | 
    ((full_flat_wave > 7160) & (full_flat_wave < 7400)) | 
    ((full_flat_wave > 8100) & (full_flat_wave < 8400))
                )
    
    full_flat_wave = full_flat_wave[telluric_mask]
    full_flat_flux = full_flat_flux[telluric_mask]
    
    return full_spectra_wave, full_spectra_flux, full_flat_wave, full_flat_flux