def binary_detect(file_name, mask_name, rv_shift_arr, t_eff, Teff_target, logg_target, met_target):
    
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks
    from util import *
    from synth_spec import *
    from ccf import *
    
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
    dRV_list = np.array([])
    Teff_list = np.array([])
    peak_list = np.array([])
    
    for i in range(0, len(rv_shift_arr)):
        for j in range(0, len(t_eff)):

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
            
            dRV_list = np.concatenate([dRV_list, rv_shift_arr[i]])
            Teff_list = np.concatenate([Teff_list, t_eff[j]])
            peak_list = np.concatenate([peak_list, peaks])
            
            np.save_txt(fname = "rv_{}_teff_{}.dat".format(rv_shift_arr[i], t_eff[j]), X=my_ccf)
            
    df = pd.DataFrame({"dRV": dRV_list, "Teff": Teff_list, "peaks": peak_list})
    
    return df
    
    
    