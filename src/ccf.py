import numpy as np

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
        return ccf, ccf_pixels