from Starfish.grid_tools import download_PHOENIX_models, \
    HDF5Interface, HDF5Creator, PHOENIXGridInterfaceNoAlpha, Interpolator
from Starfish.grid_tools.instruments import ESPaDOnS

def download_stellar_model_grid(filename, Teff_target, Z_target):
    """
    Downloads grid of PHEONIX model spectra based on the Teff and Z of the
    target star.
    Arguments:
        filename: Filename of HDF5 file in which synthetic spectra will be saved (string)
        Teff_target: Effective temperature of the target star in Kelvin (int or float)
        Z_target: Metallicity of the target star in dex (int or float)
    """
    
    ranges = [[3500, Teff_target], [4.0, 5.0], [Z_target-0.1, Z_target+0.1]]
    download_PHOENIX_models(path="PHOENIX", ranges=ranges)
    grid = PHOENIXGridInterfaceNoAlpha(path="PHOENIX")
    creator = HDF5Creator(
        grid, filename, instrument=ESPaDOnS(), wl_range=(0.45e4, 0.87e4), ranges=ranges
    )
    creator.process_grid()

def get_synth_spectrum(filename, Teff, logg, Z):
    """
    Retrieves synthetic spectrum with a given value of Teff, logg, and Z
    by interpolating PHOENIX model grid.
    Arguments:
        filename: Name of HDF5 file where model spectra are stored (string)
        Teff: Stellar effective temperature in Kelvin (int or float)
        logg: Log stellar surface gravity in dex (int or float)
        Z: Stellar metallicity in dex (int or float)
    Returns:
        wave: Wavelength of spectrum (Angstroms)
        flux: Flux density of spectrum
    """
    
    myHDF5 = HDF5Interface(filename)
    myInterpolator = Interpolator(myHDF5)
    flux = myInterpolator([Teff, logg, Z])
    wave = myHDF5.wl
    
    return wave, flux