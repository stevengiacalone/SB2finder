import numpy as np
import pandas as pd
from multiprocessing import Pool
import os
from scipy.signal import find_peaks
from util import *
from synth_spec import *
from ccf import *

def binary_detect(file_name, mask_name, rv_shift_arr, t_eff, Teff_target, logg_target, met_target):
    """
    Automated injection-recovery test.
    Args:
        file_name: KPF L1 file name (str).
        mask_name: Stellar binary mask file name (str).
        rv_shift_arr: Array of delta RVs to test (numpy array of ints or float; km/s)
        t_eff: Array of secondary effective temperatures to test (numpy array of ints or floats; K)
        Teff_target: Effective temperature of primary (int or float; K).
        logg_target: Log10 surface gravity of primary (int or float; dex).
        met_target: Metallicity of primary (int or float; dex).
    """
    
    #Get whole spectrum for observed star (KPF specific procedure)
    full_spectra_wave, full_spectra_flux, full_flat_wave, full_flat_flux = stitch_spec(file_name)
    
    line_mask = np.loadtxt(mask_name).T
    min_wave = full_flat_wave[0]
    max_wave = full_flat_wave[-1]
    line_mask_mask = (line_mask[0] > min_wave) & (line_mask[0] < max_wave) #changed mask range to 8500
    new_line_mask = line_mask[0][line_mask_mask]
    new_line_weight = line_mask[1][line_mask_mask]
    
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
    
    #download grid
    filename = "synth_spec.hdf5"
    Teff_target = Teff_target
    Z_target = met_target
    download_stellar_model_grid(filename, Teff_target, Z_target)
    
    #Get a_arr
    myHDF5 = HDF5Interface(filename)

    synth_flux1 = myHDF5.load_flux(np.array([Teff_target, logg_target, met_target]))
    synth_wave1 = myHDF5.wl
    synth_mask1 = (synth_wave1 > min_wave) & (synth_wave1 < max_wave)
    
    #prep loop variables
    dRV_list = []
    Teff_list = []
    peak_list = []
    
    for i in range(0, len(rv_shift_arr)):
        for j in range(0, len(t_eff)):
            
            print('Working on RV shift of {} km/s and T_eff of {} K'.format(rv_shift_arr[i], t_eff[j]))

            synth_flux2 = myHDF5.load_flux(np.array([t_eff[j], 4.5, 0]))
            synth_wave2 = myHDF5.wl            
            synth_mask2 = (synth_wave2 > min_wave) & (synth_wave2 < max_wave)

            _, _, _, flat_synth_flux1, yfit1 = flatspec_spline(synth_wave1[synth_mask1], synth_flux1[synth_mask1], np.ones(len(synth_wave1[synth_mask1])))
            _, _, _, flat_synth_flux2, yfit2 = flatspec_spline(synth_wave2[synth_mask2], synth_flux2[synth_mask2], np.ones(len(synth_wave2[synth_mask2])))

            a_arr = yfit1/yfit2

            combined_wave, combined_flux = combine_spectra(synth_wave2[synth_mask2], flat_synth_flux2, full_flat_wave, full_flat_flux, rv=rv_shift_arr[i], a=a_arr, vsini=2)

            test = calc_ccf(velocity_loop, new_line_start, new_line_end, combined_wave, combined_flux[1:], new_line_weight, sn, -z_b)

            mask_vel = np.ones(len(test[0]), dtype=bool)
            my_ccf = ((-test[0][mask_vel]) + np.max(test[0][mask_vel])) / np.max(((-test[0][mask_vel]) + np.max(test[0][mask_vel])))
            my_vel = velocity_loop[mask_vel]
            peaks, _ = find_peaks(my_ccf, height=0.05)
            
            dRV_list.append(rv_shift_arr[i])
            Teff_list.append(t_eff[j])
            peak_list.append(peaks)
            
            np.savetxt(fname = "rv_{}_teff_{}.dat".format(rv_shift_arr[i], t_eff[j]), X=my_ccf)
            
    df = pd.DataFrame({"dRV": dRV_list, "Teff": Teff_list, "peaks": peak_list})
    
    return df
    
def parallel_func(idx_list, dRV_list, Teff_list, peak_list, n_injections, n_cores):
    """
    Function for parallelizing injection-recovery tests.
    Args:
        idx_list: Empty list to store index values in.
        dRV_list: Empty list to store dRV values in.
        Teff_list: Empty list to store Teff values in.
        peak_list: Empty list to store peak arrays in.
        params: 2D array containing inputs.
        n_cores: Number of cores to use for parallelization.
    """
    i = np.arange(n_injections)
    if n_cores > os.cpu_count():
        n_cores = os.cpu_count()
#     with Pool(n_cores) as pool:
#         res = pool.starmap(calculate_CCF, params)
#         for r in res:
#             idx_list.append(r[0])
#             dRV_list.append(r[1])
#             Teff_list.append(r[2])
#             peak_list.append(r[3])
    with Pool(n_cores) as pool:
        res = pool.imap_unordered(calculate_CCF, i)
        for r in res:
            idx_list.append(r[0])
            dRV_list.append(r[1])
            Teff_list.append(r[2])
            peak_list.append(r[3])
        pool.close()
    return

def binary_detect_parallel(file_name, mask_name, rv_shift_arr, t_eff, Teff_target, logg_target, met_target, n_cores):
    """
    Injection-recovery tests, but parallelized. Specify number of cores to use with n_cores argument.
    """
    
    # Create results and spec directories, if they doesn't already exist
    if os.path.isdir('results') == False:
        os.mkdir("./results")
    if os.path.isdir('spec') == False:
        os.mkdir("./spec")
    
    # Get whole spectrum for observed star (KPF specific procedure)
    full_spectra_wave, full_spectra_flux, full_flat_wave, full_flat_flux = stitch_spec(file_name)
    np.savetxt("spec/flat_obs_spec.csv", np.array([full_flat_wave, full_flat_flux]), delimiter=",")
    
    # Download PHOENIX model spectra grid
    synth_file_name = "synth_spec.hdf5"
    download_stellar_model_grid(synth_file_name, Teff_target, met_target)
    
    # Get synthetic spectra file and save all to csv files
    myHDF5 = HDF5Interface(synth_file_name)
    
    # model of target star first
    min_wave = full_flat_wave[0]
    max_wave = full_flat_wave[-1]
    synth_flux1 = myHDF5.load_flux(np.array([Teff_target, logg_target, met_target]))
    synth_flux1 = synth_flux_correction(synth_flux1, Teff_target)
    synth_wave1 = myHDF5.wl
    synth_mask1 = (synth_wave1 > min_wave) & (synth_wave1 < max_wave)    
    np.savetxt("spec/target_synth_spec.csv", np.array([synth_wave1[synth_mask1], synth_flux1[synth_mask1]]), delimiter=",")

    for i in range(len(t_eff)):
        synth_flux2 = myHDF5.load_flux(np.array([t_eff[i], 4.5, met_target]))
        synth_flux2 = synth_flux_correction(synth_flux2, t_eff[i])
        synth_wave2 = myHDF5.wl            
        synth_mask2 = (synth_wave2 > min_wave) & (synth_wave2 < max_wave)
        np.savetxt(f"spec/{int(t_eff[i])}_synth_spec.csv", np.array([synth_wave2[synth_mask2], synth_flux2[synth_mask2]]), delimiter=",")
    
    # Prep loop variables
    idx_list = []
    dRV_list = []
    Teff_list = []
    peak_list = []
    
    # Combine everything into a dataframe and save it
    params = np.array(np.meshgrid(rv_shift_arr, t_eff)).T.reshape(-1, 2)
    ccf_idx = np.arange(len(params))[:, None]
    mask_name_arr = np.full_like(ccf_idx, mask_name, dtype=object)
    params = np.concatenate([mask_name_arr,
                             ccf_idx, 
                             params], axis=1, dtype=object)
    df = pd.DataFrame(params).rename(
        columns={0: "mask", 
                 1: "idx", 
                 2: "dRV", 
                 3: "Teff"}
    )
    df.to_csv("results/injections.csv")
    
    # Run parallel CCF calculation
    n_injections = len(ccf_idx)
    parallel_func(idx_list, dRV_list, Teff_list, peak_list, n_injections, n_cores)        
            
    # Save results
    df_res = pd.DataFrame({"idx": idx_list, "dRV": dRV_list, "Teff": Teff_list, "peaks": peak_list})
    df_res.to_csv("results/injrec_results.csv")
    
    return df