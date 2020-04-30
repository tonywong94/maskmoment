import numpy as np
import os
from spectral_cube import SpectralCube
from astropy.io import fits
from astropy import units as u
from momfuncs import makenoise, dilmsk, smcube, findflux

#from datetime import datetime
#from astropy.stats import mad_std
#from scipy import ndimage
#import radio_beam
#from astropy.convolution import Gaussian1DKernel
#from astropy import wcs
#from astropy.convolution import convolve_fft

# 1. Need to calculate errors in moments
# 2. Need to allow different spectral smoothing (e.g. boxcar)

def maskmoment(img_fits, gain_fits=None, rms_fits=None, outdir=None, outname=None, 
                snr_hi=4, snr_lo=2, minbeam=1, min_thresh_ch=1, min_tot_ch=2, 
                nguard=[0,0], edgech=5, fwhm=None, vsm=None, vsm_type='gauss',
                output_snr_cube=False, to_kelvin=True):
    """
    Produce FITS images of moment maps using a dilated mask approach.

    Parameters
    ----------
    img_fits : FITS file name, required
        The image cube, this should be in units of K, Jy/beam, or equivalent.
    gain_fits : FITS file name, optional
        The gain cube, e.g. pb cube from CASA.  This should have a value between
        0 and 1, with 0 near the edges and 1 near the center of the image, and 
        have the same dimensions as the image cube.
        NOTE: The gain cube is ignored if a noise cube (rms_fits) is given.
    rms_fits : FITS file name, optional
        The noise cube, providing an estimate of the rms noise at each pixel.
        This should have the same dimensions and units as the image cube.
        NOTE: If rms_fits is not given, a noise cube is generated from the
        image cube, after removing any gain variation using the gain cube.
    outdir : string, optional
        Directory to write the output files
        Default: Same directory as img_fits.
        NOTE: Currently this directory is assumed to exist.
    outname : string, optional
        Basname for output files.  For instance, outname='foo' produces files
        'foo.mom0.fits.gz', etc.
        Default: Based on root name of img_fits.
    snr_hi : float, optional
        The high significance threshold from which to begin the mask dilation.
        Default: 4
    snr_lo : float, optional
        The low significance threshold at which to end the mask dilation.
        Default: 2
    minbeam : float, optional
        Minimum velocity-integrated area of a mask region in units of the beam size.
        Default: 1
    min_thresh_ch : int, optional
        High significance mask is required to span at least this many channels
        at all pixels.
        Default: 1
    min_tot_ch : int, optional
        Dilated mask regions are required to span at least this many channels.
        Default: 2
    nguard : tuple of two ints, optional
        Expand the final mask by this nguard[0] pixels in sky directions and
        nguard[1] channels in velocity.  Currently these values must be equal
        if both are non-zero.
        If nguard[0] = 0 then no expansion is done in sky coordinates.
        If nguard[1] = 0 then no expansion is done in velocity.
        Default: [0,0]
    edgech : int, optional
        Number of channels at each end of cube to use for rms estimation.
        Default: 5
    fwhm : float or :class:`~astropy.units.Quantity`, optional
        Spatial resolution to smooth to before generating the dilated mask.  
        If value is not astropy quantity, assumed to be given in arcsec.
        Default: No spatial smoothing is applied.
    vsm : float or :class:`~astropy.units.Quantity`, optional
        Width of the spectral smoothing kernel.  If given as astropy quantity,
        should be given in velocity units.  If not given as astropy quantity, 
        assumed to be given in *numbers of channels*.
        Default: No spectral smoothing is applied.
    vsm_type : string, optional
        What type of spectral smoothing to employ.  Currently three options:
        (1) 'boxcar' - 1D boxcar smoothing, vsm rounded to integer no. of chans.
        (2) 'gauss' - 1D gaussian smoothing, vsm is the convolving gaussian sigma.
        (3) 'gaussfinal' - 1D gaussian smoothing, vsm is the gaussian sigma
        after convolution, assuming sigma before convolution is 1 channel width.        
        Default: 'gauss'
    output_snr_cube : boolean, optional
        Output the cube is SNR units in addition to the moment maps.
        Default: False
    to_kelvin : boolean, optional
        Output the moment maps in K units if the cube is in Jy/beam units.
        Default: True
    """
    np.seterr(divide='ignore', invalid='ignore')
    # --- Determine output file names
    if outdir is not None:
        pth = outdir+'/'
    else:
        pth = os.path.dirname(img_fits)+'/'
    if outname is not None:
        basename = outname
    else:
        if img_fits.endswith('.fits.gz'):
            basename = os.path.splitext(os.path.splitext(img_fits)[0])[0]
        else:
            basename = os.path.splitext(img_fits)[0]
    print('\nOutput basename is:',basename)
    # --- Read input files
    image_cube = SpectralCube.read(img_fits)
    hdr = image_cube.header
    savehd = hdr.copy()
    print('Image cube '+img_fits+':\n',image_cube)
    if rms_fits is not None:
        rms_cube = SpectralCube.read(rms_fits)
        print('Noise cube '+rms_fits+':\n',rms_cube)
        if rms_cube.unit != image_cube.unit:
            raise RuntimeError('Noise cube units', rms_cube.unit, 
                               'differ from image units', image_cube.unit)
    elif gain_fits is not None:
        gain_cube  = SpectralCube.read(gain_fits)
        print('Gain cube '+gain_fits+':\n',gain_cube)
        rms_cube = makenoise(image_cube, gain_cube, edge=edgech)
        print('Noise cube:\n',rms_cube)
        hdr['datamin'] = np.nanmin(rms_cube)
        hdr['datamax'] = np.nanmax(rms_cube)
        fits.writeto(pth+basename+'.ecube.fits.gz', rms_cube._data.astype(np.float32),
                 hdr, overwrite=True)
        print('Wrote', pth+basename+'.ecube.fits.gz')
    # --- Peak SNR image
    snr_cube = image_cube / rms_cube
    print('\nSNR cube:\n',snr_cube)
    if output_snr_cube:
        hdr['datamin'] = snr_cube.min().value
        hdr['datamax'] = snr_cube.max().value
        hdr['bunit'] = ' '
        fits.writeto(pth+basename+'.snrcube.fits.gz', snr_cube._data.astype(np.float32),
                     hdr, overwrite=True)
        print('Wrote', pth+basename+'.snrcube.fits.gz')    
    snr_peak = snr_cube.max(axis=0)
    hd2d = hdr.copy()
    hd2d['datamin'] = np.nanmin(snr_peak.value)
    hd2d['datamax'] = np.nanmax(snr_peak.value)
    hd2d['bunit'] = ' '
    fits.writeto(pth+basename+'.snrpk.fits.gz', snr_peak.astype(np.float32),
                 hd2d, overwrite=True)
    print('Wrote', pth+basename+'.snrpk.fits.gz')
    # --- Generate dilated mask
    if fwhm > 0:
        sm_snrcube = smcube(snr_cube, fwhm=fwhm, vsm=vsm, edgech=edgech)
        print('Smoothed SNR cube:\n', sm_snrcube)
        dilcube = sm_snrcube
    else:
        dilcube = snr_cube
    dilatedmask = dilmsk(dilcube, header=hdr, snr_hi=snr_hi, snr_lo=snr_lo, 
                         min_thresh_ch=min_thresh_ch, min_tot_ch=min_tot_ch, 
                         nguard=nguard, minbeam=minbeam, debug=False)
    hdr['datamin'] = 0
    hdr['datamax'] = 1
    hdr['bunit'] = ' '
    fits.writeto(pth+basename+'.mask.fits.gz', dilatedmask.astype(np.float32),
                 hdr, overwrite=True)
    print('Wrote', pth+basename+'.mask.fits.gz')
    nchanimg = np.sum(dilatedmask, axis=0)
    # --- Generate moment maps from masked cube
    dil_mskcub = image_cube.with_mask(dilatedmask > 0)
    print('Units of cube are', dil_mskcub.unit)
    if to_kelvin:
        dil_mskcub_k = dil_mskcub.to(u.K)
        dil_mskcub = dil_mskcub_k
        dil_mskcub_mom0 = dil_mskcub.moment(order=0).to(u.K*u.km/u.s)
    elif all(x in (savehd['bunit']).upper() for x in ['JY', 'B']):  # keep Jy/bm
        #omega_B = ((savehd['bmaj']*u.deg*savehd['bmin']*u.deg)*2*np.pi/(8*np.log(2)))
        dil_mskcub_mom0 = dil_mskcub.moment(order=0).to(u.Jy/u.beam*u.km/u.s)
    else:
        dil_mskcub_mom0 = dil_mskcub.moment(order=0)
    if hasattr(dil_mskcub_mom0, 'unit'):
        print('Units of mom0 map are', dil_mskcub_mom0.unit)
    hdr['datamin'] = np.nanmin(dil_mskcub_mom0.value)
    hdr['datamax'] = np.nanmax(dil_mskcub_mom0.value)
    hdr['bunit'] = dil_mskcub_mom0.unit.to_string('fits')
    fits.writeto(pth+basename+'.mom0.fits.gz', dil_mskcub_mom0.astype(np.float32),
                 hdr, overwrite=True)
    print('Wrote', pth+basename+'.mom0.fits.gz')
    # --- Moment 1: require at least 2 unmasked channels at each pixel
    dil_mskcub_mom1 = dil_mskcub.moment(order=1).to(u.km/u.s)
    vmin = dil_mskcub.spectral_extrema[0]
    vmax = dil_mskcub.spectral_extrema[1]
    dil_mskcub_mom1[dil_mskcub_mom1 < vmin] = np.nan
    dil_mskcub_mom1[dil_mskcub_mom1 > vmax] = np.nan
    dil_mskcub_mom1[nchanimg < 2] = np.nan
    hdr['datamin'] = np.nanmin(dil_mskcub_mom1.value)
    hdr['datamax'] = np.nanmax(dil_mskcub_mom1.value)
    hdr['bunit'] = dil_mskcub_mom1.unit.to_string('fits')
    fits.writeto(pth+basename+'.mom1.fits.gz', dil_mskcub_mom1.astype(np.float32),
                 hdr, overwrite=True)
    print('Wrote', pth+basename+'.mom1.fits.gz')
    # --- Moment 2: require at least 3 unmasked channels at each pixel
    dil_mskcub_mom2 = dil_mskcub.linewidth_sigma().to(u.km/u.s)
    dil_mskcub_mom2[nchanimg < 3] = np.nan
    hdr['datamin'] = np.nanmin(dil_mskcub_mom2.value)
    hdr['datamax'] = np.nanmax(dil_mskcub_mom2.value)
    hdr['bunit'] = dil_mskcub_mom2.unit.to_string('fits')
    fits.writeto(pth+basename+'.mom2.fits.gz', dil_mskcub_mom2.astype(np.float32),
                 hdr, overwrite=True)
    print('Wrote', pth+basename+'.mom2.fits.gz')
    fluxtab = findflux(image_cube, rms_cube, dilatedmask)
    fluxtab.write(pth+basename+'.flux.csv', delimiter=',', format='ascii.ecsv', 
                  overwrite=True)
    print('Wrote', pth+basename+'.flux.csv')
    return

# def moments_smo(img_fits, gain_fits=None, rms_fits=None, outdir=None, outname=None, 
#                 snr_hi=4, snr_lo=2, min_thresh_ch=1, min_tot_ch=2, nguard=[0,0], 
#                 minbeam=1, edgech=5, fwhm=12, vsm=None, output_snr_cube=False):
#     np.seterr(divide='ignore', invalid='ignore')
#     # --- Determine output file names
#     if outdir is not None:
#         pth = outdir+'/'
#     else:
#         pth = os.path.dirname(img_fits)+'/'
#     if outname is not None:
#         basename = outname
#     else:
#         if img_fits.endswith('.fits.gz'):
#             basename = os.path.splitext(os.path.splitext(img_fits)[0])[0]
#         else:
#             basename = os.path.splitext(img_fits)[0]
#     print('\nOutput basename is:',basename)
#     # --- Read input files
#     image_cube = SpectralCube.read(img_fits)
#     hdr = image_cube.header
#     print('Image cube '+img_fits+':\n',image_cube)
#     if rms_fits is not None:
#         rms_cube = SpectralCube.read(rms_fits)
#         print('Noise cube '+rms_fits+':\n',rms_cube)
#         # Should check that units of rms_cube match image_cube
#     elif gain_fits is not None:
#         gain_cube  = SpectralCube.read(gain_fits)
#         print('Gain cube '+gain_fits+':\n',gain_cube)
#         rms_cube = makenoise(image_cube, gain_cube, edge=edgech)
#         print('Noise cube:\n',rms_cube)
#         hdr['datamin'] = np.nanmin(rms_cube)
#         hdr['datamax'] = np.nanmax(rms_cube)
#         fits.writeto(pth+basename+'.ecube.fits.gz', rms_cube._data.astype(np.float32),
#                  hdr, overwrite=True)
#         print('Wrote', pth+basename+'.ecube.fits.gz')
#     # --- Peak SNR image
#     snr_cube = image_cube / rms_cube
#     snr_peak = snr_cube.max(axis=0)
#     hd2d = hdr.copy()
#     hd2d['datamin'] = np.nanmin(snr_peak.value)
#     hd2d['datamax'] = np.nanmax(snr_peak.value)
#     hd2d['bunit'] = ' '
#     fits.writeto(pth+basename+'.snrpk.fits.gz', snr_peak.astype(np.float32),
#                  hd2d, overwrite=True)
#     print('Wrote', pth+basename+'.snrpk.fits.gz')
#     if output_snr_cube:
#         hdr['datamin'] = snr_cube.min().value
#         hdr['datamax'] = snr_cube.max().value
#         hdr['bunit'] = ' '
#         fits.writeto(pth+basename+'.snrcube.fits.gz', snr_cube._data.astype(np.float32),
#                      hdr, overwrite=True)
#         print('Wrote', pth+basename+'.snrcube.fits.gz')    
#     # --- Generate dilated mask
#     sm_snrcube = smcube(snr_cube, fwhm=fwhm, vsm=vsm, edgech=edgech)
#     print(sm_snrcube)
#     dilatedmask = dilmsk(sm_snrcube, header=hdr, snr_hi=snr_hi, snr_lo=snr_lo, 
#                          min_thresh_ch=min_thresh_ch, min_tot_ch=min_tot_ch, 
#                          nguard=nguard, minbeam=minbeam, debug=False)
#     hdr['datamin'] = 0
#     hdr['datamax'] = 1
#     hdr['bunit'] = ' '
#     fits.writeto(pth+basename+'.mask.fits.gz', dilatedmask.astype(np.float32),
#                  hdr, overwrite=True)
#     print('Wrote', pth+basename+'.mask.fits.gz')
#     return