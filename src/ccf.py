import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from util import *

LIGHT_SPEED = 3e5

def calc_ccf(velocity_loop, new_line_start, new_line_end, x_pixel_wave, spectrum, new_line_weight, sn, zb=0):
        """ Cross correlation by the shifted mask line and the spectrum data of one order for each velocity step.
        Code from KPF pipeline: https://github.com/Keck-DataReductionPipelines/KPF-Pipeline

        Args:
            velocity_loop (array): Velocity steps.
            new_line_start (numpy.ndarray): Start of the mask line.
            new_line_end (numpy.ndarray): End of the mask line.
            x_pixel_wave (numpy.ndarray): Wavelength calibration of the pixels.
            spectrum (numpy.ndarray): 1D Spectrum data.
            new_line_weight (numpy.ndarray): Mask weight
            sn (numpy.ndarray): Additional SNR scaling factor (comply with the implementation of CCF of C version)
            zb (float): Redshift at the observation time.

        Returns:
            numpy.ndarray: ccf at velocity steps.
            numpy.ndarray: Intermediate CCF numbers at pixels.
        """

        v_steps = len(velocity_loop)
        ccf = np.zeros(v_steps)
        shift_lines_by = (1.0 + (velocity_loop / LIGHT_SPEED)) / (1.0 + zb)

        n_pixel = np.shape(x_pixel_wave)[0] - 1                # total size in  x_pixel_wave_start
        n_line_index = np.shape(new_line_start)[0]

        pix1, pix2 = 0, n_pixel-1
        x_pixel_wave_end = x_pixel_wave[1: n_pixel+1]            # total size: n_pixel
        x_pixel_wave_start = x_pixel_wave[0: n_pixel]
        ccf_pixels = np.zeros([v_steps, n_pixel])

        counter = 0
        for c in range(v_steps):
            line_doppler_shifted_start = new_line_start * shift_lines_by[c]
            line_doppler_shifted_end = new_line_end * shift_lines_by[c]

            closest_match = np.sum((x_pixel_wave_end - line_doppler_shifted_start[:, np.newaxis] < 0.), axis=1)
            closest_match_next = np.sum((x_pixel_wave_start - line_doppler_shifted_end[:, np.newaxis] <= 0.), axis=1)
            mask_spectra_doppler_shifted = np.zeros(n_pixel)

            idx_collection = list()
            for k in range(n_line_index):
                closest_x_pixel = closest_match[k]  # closest index starting before line_dopplershifted_start
                closest_x_pixel_next = closest_match_next[k]  # closest index starting after line_dopplershifted_end
                line_start_wave = line_doppler_shifted_start[k]
                line_end_wave = line_doppler_shifted_end[k]
                line_weight = new_line_weight[k]

                if closest_x_pixel_next <= pix1 or closest_x_pixel >= pix2:
                    continue
                else:
                    for n in range(closest_x_pixel, closest_x_pixel_next):
                        if n > pix2:
                            break
                        if n < pix1:
                            continue
                        # if there is overlap
                        if x_pixel_wave_start[n] <= line_end_wave and x_pixel_wave_end[n] >= line_start_wave:
                            wave_start = max(x_pixel_wave_start[n], line_start_wave)
                            wave_end = min(x_pixel_wave_end[n], line_end_wave)
                            mask_spectra_doppler_shifted[n] = line_weight * (wave_end - wave_start) / \
                                (x_pixel_wave_end[n] - x_pixel_wave_start[n])

                            if n in idx_collection:
                                pass
                                # print(str(n), ' already taken')
                            else:
                                idx_collection.append(n)
            ccf_pixels[c, :] = spectrum * mask_spectra_doppler_shifted * sn
            ccf[c] = np.nansum(ccf_pixels[c, :])

            # print(counter)
            counter += 1

        return ccf, ccf_pixels
    
def calculate_CCF(i):
    
    df = pd.read_csv("results/injections.csv")
    mask_name = df["mask"].values[i]
    idx = df["idx"].values[i]
    dRV = df["dRV"].values[i]
    Teff = df["Teff"].values[i]
    print(f"Working on RV shift of {dRV} km/s and T_eff of {Teff} K")
    
    # Get whole spectrum for observed star (KPF specific procedure)
    full_flat_wave, full_flat_flux = np.loadtxt("spec/flat_obs_spec.csv", delimiter=",")
    
    line_mask = np.loadtxt(mask_name).T
    min_wave = full_flat_wave[0]
    max_wave = full_flat_wave[-1]
    line_mask_mask = (line_mask[0] > min_wave) & (line_mask[0] < max_wave)
    new_line_mask = line_mask[0][line_mask_mask]
    new_line_weight = line_mask[1][line_mask_mask]
    
    # Set up CCF inputs
    velocity_loop = np.arange(-50, 50, 1.0) # velocities to calculate CCF at
    v_steps = len(velocity_loop)
    LIGHT_SPEED = 3e5
    LIGHT_SPEED_M = 3e8
    vb = -82e3
    zb = vb/LIGHT_SPEED_M
    z_b = ((1.0/(1+zb)) - 1.0)

    new_line_start = new_line_mask - 0.025
    new_line_end = new_line_mask + 0.025
    x_pixel_wave = full_flat_wave
    spectrum = full_flat_flux[1:]
    sn = np.ones(len(spectrum))
    
    # Get appropriate synthetic spectra 
    synth_wave1, synth_flux1 = np.loadtxt("spec/target_synth_spec.csv", delimiter=",")
    synth_wave2, synth_flux2 = np.loadtxt(f"spec/{int(Teff)}_synth_spec.csv", delimiter=",")

    _, _, _, flat_synth_flux1, yfit1 = flatspec_spline(
        synth_wave1, 
        synth_flux1, 
        np.ones(len(synth_wave1)),
        window=100
    )
    _, _, _, flat_synth_flux2, yfit2 = flatspec_spline(
        synth_wave2, 
        synth_flux2, 
        np.ones(len(synth_wave2)),
        window=100
    )

    a_arr = yfit1/yfit2

    # Combine the secondary spectrum with the observed spectrum (82 km/s added to account for RV shift in KPF data)
    combined_wave, combined_flux = combine_spectra(
        synth_wave2, 
        flat_synth_flux2, 
        full_flat_wave, 
        full_flat_flux, 
        rv=dRV+82, 
        a=a_arr, 
        vsini=2
    )

    # Calculate the ccf, normalize it, and search for peaks
    ccf = calc_ccf(
        velocity_loop, 
        new_line_start, 
        new_line_end, 
        combined_wave, 
        combined_flux[1:], 
        new_line_weight, 
        sn, 
        -z_b
    )

    norm_ccf = ((-ccf[0]) + np.max(ccf[0])) / np.max(((-ccf[0]) + np.max(ccf[0])))
    peaks, _ = find_peaks(norm_ccf, height=0.05)

    # save ccf
    np.savetxt(fname = f"results/rv_{dRV}_teff_{Teff}.dat", X=norm_ccf)
    
    return idx, dRV, Teff, peaks