import os
import numpy as np
from spectral_cube import SpectralCube
from spectral_cube.cube_utils import convert_bunit
from astropy.io import fits
from astropy import units as u
from astropy import wcs
from .momfuncs import makenoise, dilmsk, smcube, findflux, writemom, calc_moments


def maskmoment(img_fits, gain_fits=None, rms_fits=None, mask_fits=None, outdir='', 
                outname=None, snr_hi=4, snr_lo=2, minbeam=1, snr_hi_minch=1, 
                snr_lo_minch=1, min_tot_ch=2, nguard=[0,0], edgech=5, fwhm=None, 
                vsm=None, vsm_type='gauss', mom1_minch=2, mom2_minch=2, altoutput=False, 
                output_peak=False, output_snr_cube=False, output_snr_peak=False, 
                output_snrsm_cube=False, output_2d_mask=False, to_kelvin=True, 
                huge_operations=True, perpixel=False, splitchan=None, rms=None,
                vmin=None, vmax=None):
    """
    Produce FITS images of moment maps using a dilated masking approach.

    Parameters
    ----------
    img_fits : FITS file name, required
        The image cube, this should be in units of K, Jy/beam, or equivalent.
    gain_fits : FITS file name, optional
        The gain cube, e.g. pb cube from CASA.  This should have a value between
        0 and 1, with 0 near the edges and 1 near the center of the image, and 
        have the same dimensions as (or broadcastable to) the image cube.
        NOTE: The gain cube is ignored if a noise cube (rms_fits) is given.
    rms_fits : FITS file name, optional
        The noise cube, providing an estimate of the rms noise at each pixel.
        This should have the same dimensions and units as the image cube.
        If it is a 2D array it will be replicated along the velocity axis.
        NOTE: If rms_fits is not given, a noise cube is generated from the
        image cube, taking into account any gain variation using the gain cube.
    rms : float, optional
        Global estimate of the rms noise, to be used instead of trying
        to estimate it from the data.  It should have the units of the input cube.
        Default is to use rms_fits, or to calculate the rms from the data.
    mask_fits : FITS file name, optional
        External mask cube to use.  This cube should have 1's for valid pixels 
        and 0's for excluded pixels.  If this is provided then the mask generation
        is skipped and the program goes straight to calculating the moments.
    outdir : string, optional
        Directory to write the output files.
        Default: Write to the directory where img_fits resides.
    outname : string, optional
        Basename for output files.  For instance, outname='foo' produces files
        'foo.mom0.fits.gz', etc.
        Default: Based on root name of img_fits.
    snr_hi : float, optional
        The high significance sigma threshold from which to begin mask dilation.
        Use a very low number (such as -10) to produce unmasked moment maps.
        Default: 4
    snr_lo : float, optional
        The low significance sigma threshold at which to end mask dilation.
        Default: 2
    snr_hi_minch : int, optional
        High significance mask is required to span at least this many channels
        at all pixels.
        Default: 1
    snr_lo_minch : int, optional
        Low significance mask is required to span at least this many channels
        at all pixels.
        Default: 1
    min_tot_ch : int, optional
        Each contiguous mask region must span at least this many channels (but 
        it's not necessary that every pixel in the region span this many channels).
        Default: 2
    minbeam : float, optional
        Minimum velocity-integrated area of a mask region in units of the beam size.
        Default: 1
    nguard : tuple of two ints, optional
        Expand the final mask by nguard[0] pixels in the sky directions and
        nguard[1] channels in velocity.  Currently these values must be equal
        if both are non-zero.
        If nguard[0] = 0 then no expansion is done in sky coordinates.
        If nguard[1] = 0 then no expansion is done in velocity.
        Default: [0,0]
    edgech : int, optional
        Number of channels at each end of vel axis to use for rms estimation.
        Default: 5
    fwhm : float or :class:`~astropy.units.Quantity`, optional
        Spatial resolution to smooth to before generating the dilated mask.  
        If value is not an astropy quantity, assumed to be given in arcsec.
        Default: No spatial smoothing is applied.
    vsm : float or :class:`~astropy.units.Quantity`, optional
        Full width of the spectral smoothing kernel (or FWHM for gaussian).  
        If given as astropy quantity, should be provided in velocity units.  
        If NOT given as astropy quantity, interpreted as number of channels.
        Default: No spectral smoothing is applied.
    vsm_type : string, optional
        What type of spectral smoothing to employ.  Currently three options:
        (1) 'boxcar' - 1D boxcar smoothing, vsm rounded to integer # of chans.
        (2) 'gauss' - 1D gaussian smoothing, vsm is the convolving gaussian FWHM.
        (3) 'gaussfinal' - 1D gaussian smoothing, vsm is the gaussian FWHM
            after convolution, assuming FWHM before convolution is 1 channel.        
        Default: 'gauss'
    mom1_minch : int, optional
        Minimum number of unmasked channels needed to calculate moment-1.
        Default: 2
    mom2_minch : int, optional
        Minimum number of unmasked channels needed to calculate moment-2.
        Default: 2
    perpixel : boolean, optional
        Whether to calculate the rms per XY pixel instead of over whole image.
        Set to True if you know there is a sensitivity variation across the image
        but you don't have a gain cube - requires rms_fits and gain_fits unset.
        Default: False
    output_peak : boolean, optional
        Output the peak brightness, velocity at peak brightness, and effective 
        line width (mom0/peak) in addition to the moment maps.  The peak brightness
        has no mask applied, the other two are derived from the masked cube.  The 
        effective line width is normalized to match mom-2 for a pure Gaussian profile.
        Default: False
    output_snr_cube : boolean, optional
        Output the (unmasked) cube in SNR units in addition to the moment maps.
        Default: False
    output_snr_peak : boolean, optional
        Output the (unmasked) peak SNR image in addition to the moment maps.
        Default: False
    output_snrsm_cube : boolean, optional
        Output the smoothed cube in SNR units in addition to the moment maps.
        Default: False
    output_2d_mask : boolean, optional
        Output the projected 2-D mask (replicated along the velocity axis)
        as well as the newly generated mask.
        The projected mask at a given pixel is valid for all channels as
        long as the parent mask is valid for any channel.
        Default: False
    to_kelvin : boolean, optional
        Output the moment maps in K units if the cube is in Jy/beam units.
        Default: True
    altoutput : boolean, optional
        Also output moment maps from a "direct" calculation instead of
        the moment method in spectral_cube.  Mainly used for debugging.
        Default: False
    splitchan : int, optional
        If there is a discontinuity in the spectrum noise, splitchan can be used
        to specify the channel number (counting from 0) where the 2nd section starts.
        An estimate of the noise will be determined for each section separately.
    huge_operations : boolean, optional
        Allow huge operations in spectral_cube.
        Default: True
    vmin : float or :class:`~astropy.units.Quantity`, optional
        Minimum channel number or velocity to use for all calculations.  Note
        that channels are discarded before any rms estimation using edgech.
        If given as astropy quantity, should be provided in velocity units.  
        If NOT given as astropy quantity, interpreted as channel number.
        Default: None.
    vmax : float or :class:`~astropy.units.Quantity`, optional
        Maximum channel number or velocity to use for all calculations.  Note
        that channels are discarded before any rms estimation using edgech.
        If given as astropy quantity, should be provided in velocity units.  
        If NOT given as astropy quantity, interpreted as channel number.
        Default: None.
    """
    np.seterr(divide='ignore', invalid='ignore')
    #
    # --- DETERMINE OUTPUT FILE NAMES
    #
    if outdir != '':
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
            print('\nCreated directory',outdir)
        pth = os.path.join(outdir, '')
    else:
        img_dir = os.path.dirname(img_fits)
        pth = os.path.join(img_dir, '')
    if outname is not None:
        basename = outname
    else:
        basename = os.path.basename(img_fits)
        if basename.endswith('.fits.gz'):
            basename = os.path.splitext(os.path.splitext(basename)[0])[0]
        else:
            basename = os.path.splitext(basename)[0]
    print('\nOutput basename is:',basename)
    #
    # --- READ INPUT FILES, OUTPUT NOISE CUBE IF NEWLY GENERATED
    #
    # -- Read the image cube
    image_cube = SpectralCube.read(img_fits)
    hd3d = image_cube.header
    if hd3d['CUNIT3'] == 'Hz':
        print('Image cube in frequency '+img_fits+':\n',image_cube)
        image_cube = image_cube.with_spectral_unit(u.m / u.s, 'radio')
    hd2d = hd3d.copy()
    hd2d['WCSAXES'] = 2
    for key in ['CRVAL3', 'CTYPE3', 'CRPIX3', 'CDELT3', 'CUNIT3']:
        del hd2d[key]
    has_jypbeam = all(x in (hd3d['bunit']).upper() for x in ['JY', 'B'])
    print('Image cube '+img_fits+':\n',image_cube)
    # -- Apply velocity slicing if requested
    v0 = image_cube.spectral_axis[0]
    v1 = image_cube.spectral_axis[-1]
    if (v1 < v0):
        v0, v1 = v1, v0
    if vmin is not None:
        if hasattr(vmin, 'unit'):
            v0 = vmin
        else:
            v0 = image_cube.spectral_axis[vmin]
    if vmax is not None:
        if hasattr(vmax, 'unit'):
            v1 = vmax
        else:
            v1 = image_cube.spectral_axis[vmax]
    if vmin is not None or vmax is not None:
        print('Trimming cube to velocity range of {} to {}'.format(v0,v1))
        image_cube = image_cube.spectral_slab(v0, v1)
        print('Trimmed cube '+img_fits+':\n',image_cube)
        hd3d = image_cube.header
    # -- Read the noise cube if given
    if rms_fits is not None:
        rms_dat, rms_hd = fits.getdata(rms_fits, header=True)
        unit = convert_bunit(rms_hd['bunit'])
        if rms_hd['naxis'] == 2:
            print('INFO: The rms is provided as a 2-d map')
            rmsarray = np.broadcast_to(np.expand_dims(rms_dat, axis=0), image_cube.shape)
            rms_cube = SpectralCube(data=rmsarray*unit, header=hd3d, wcs=wcs.WCS(hd3d))
        elif rms_dat.shape[0] == 1:
            print('INFO: The rms cube has a degenerate velocity axis')
            rmsarray = np.broadcast_to(rms_dat, image_cube.shape)
            rms_cube = SpectralCube(data=rmsarray*unit, header=hd3d, wcs=wcs.WCS(hd3d))
            print(rms_cube)
        else:            
            rms_cube = SpectralCube.read(rms_fits)
            if rms_cube.header['CUNIT3'] == 'Hz':
                rms_cube = rms_cube.with_spectral_unit(u.m / u.s, 'radio')
            if vmin is not None or vmax is not None:
                rms_cube = rms_cube.spectral_slab(v0, v1)
        if rms_cube.unit != image_cube.unit:
            raise RuntimeError('Noise cube units', rms_cube.unit, 
                               'differ from image units', image_cube.unit)
        print('Noise cube '+rms_fits+':\n',rms_cube)
    else:
    # -- or read the gain cube
        if gain_fits is not None:
            try:
                gain_cube  = SpectralCube.read(gain_fits)
                if gain_cube.header['CUNIT3'] == 'Hz':
                    gain_cube = gain_cube.with_spectral_unit(u.m / u.s, 'radio')
                if vmin is not None or vmax is not None:
                    gain_cube = gain_cube.spectral_slab(v0, v1)
                print('Gain cube '+gain_fits+':\n',gain_cube)
            except ValueError:
                gain_cube = fits.getdata(gain_fits)
                print('INFO: Gain is not a cube - reading as ndarray')
            rms_cube = makenoise(image_cube, gain_cube, edge=edgech, rms=rms,
                                 splitchan=splitchan)
        else:
            rms_cube = makenoise(image_cube, edge=edgech, perpixel=perpixel, rms=rms,
                                 splitchan=splitchan)
        print('Noise cube:\n',rms_cube)
        hd3d['datamin'] = np.nanmin(rms_cube._data)
        hd3d['datamax'] = np.nanmax(rms_cube._data)
        fits.writeto(pth+basename+'.ecube.fits.gz', rms_cube._data.astype(np.float32),
                 hd3d, overwrite=True)
        print('Wrote', pth+basename+'.ecube.fits.gz')
    if huge_operations:
        image_cube.allow_huge_operations = True
        rms_cube.allow_huge_operations = True
    #
    # --- GENERATE AND OUTPUT SNR CUBE, PEAK SNR IMAGE
    #
    snr_cube = image_cube / rms_cube
    print('SNR cube:\n',snr_cube)
    if output_snr_cube:
        hd3d['datamin'] = snr_cube.min().value
        hd3d['datamax'] = snr_cube.max().value
        hd3d['bunit'] = ' '
        fits.writeto(pth+basename+'.snrcube.fits.gz', snr_cube._data.astype(np.float32),
                     hd3d, overwrite=True)
        print('Wrote', pth+basename+'.snrcube.fits.gz')    
    if output_snr_peak:
        snr_peak = snr_cube.max(axis=0)
        hd2d['datamin'] = np.nanmin(snr_peak.value)
        hd2d['datamax'] = np.nanmax(snr_peak.value)
        hd2d['bunit'] = ' '
        fits.writeto(pth+basename+'.snrpk.fits.gz', snr_peak.astype(np.float32),
                     hd2d, overwrite=True)
        print('Wrote', pth+basename+'.snrpk.fits.gz')
    #
    # --- GENERATE AND OUTPUT DILATED MASK
    #
    if mask_fits is not None:
        dilatedmask = fits.getdata(mask_fits)
    else:
        if fwhm is not None or vsm is not None:
            sm_snrcube = smcube(snr_cube, fwhm=fwhm, vsm=vsm, vsm_type=vsm_type, 
                                edgech=edgech, huge=huge_operations)
            print('Smoothed SNR cube:\n', sm_snrcube)
            if output_snrsm_cube:
                hd3d['datamin'] = sm_snrcube.min().value
                hd3d['datamax'] = sm_snrcube.max().value
                hd3d['bunit'] = ' '
                fits.writeto(pth+basename+'.snrsmcube.fits.gz', 
                             sm_snrcube._data.astype(np.float32),
                             hd3d, overwrite=True)
                print('Wrote', pth+basename+'.snrsmcube.fits.gz')    
            dilcube = sm_snrcube
        else:
            dilcube = snr_cube
        dilatedmask = dilmsk(dilcube, header=hd3d, snr_hi=snr_hi, snr_lo=snr_lo, 
                             snr_hi_minch=snr_hi_minch, snr_lo_minch=snr_lo_minch, 
                             min_tot_ch=min_tot_ch, nguard=nguard, minbeam=minbeam)
        hd3d['datamin'] = 0
        hd3d['datamax'] = 1
        hd3d['bunit'] = ' '
        fits.writeto(pth+basename+'.mask.fits.gz', dilatedmask.astype(np.float32),
                     hd3d, overwrite=True)
        print('Wrote', pth+basename+'.mask.fits.gz')
        if output_2d_mask:
            summed_msk = np.broadcast_to(np.sum(dilatedmask, axis=0), image_cube.shape)
            proj_msk = np.minimum(summed_msk, 1)
            fits.writeto(pth+basename+'.mask2d.fits.gz', proj_msk.astype(np.float32),
                         hd3d, overwrite=True)
            print('Wrote', pth+basename+'.mask2d.fits.gz')
    #
    # --- CALCULATE FLUXES (IN ORIGINAL UNITS)
    #
    if mask_fits is None and output_2d_mask:
        fluxtab = findflux(image_cube, rms_cube, dilatedmask, proj_msk)
    else:
        fluxtab = findflux(image_cube, rms_cube, dilatedmask)
    fluxtab.write(pth+basename+'.flux.csv', delimiter=',', format='ascii.ecsv', 
                  overwrite=True)
    print('Wrote', pth+basename+'.flux.csv')
    #
    # --- GENERATE AND OUTPUT MOMENT MAPS
    #
    nchanimg = np.sum(dilatedmask, axis=0)
    print('Units of cube are', image_cube.unit)
    if to_kelvin and has_jypbeam:
        if hasattr(image_cube, 'beam'):
            print('Beam info:', image_cube.beam)
        else:
            print('WARNING: Beam info is missing')
        image_cube = image_cube.to(u.K)
        rms_cube = rms_cube.to(u.K)
    # --- Moment 0
    dil_mskcub = image_cube.with_mask(dilatedmask > 0)
    dil_mskcub_mom0 = dil_mskcub.moment(order=0).to(image_cube.unit*u.km/u.s)
    if hasattr(dil_mskcub_mom0, 'unit'):
        print('Units of mom0 map are', dil_mskcub_mom0.unit)
    writemom(dil_mskcub_mom0, type='mom0', filename=pth+basename, hdr=hd2d)
    # --- Peak brightness and effective linewidth (optional)
    if output_peak:
        peak = image_cube.max(axis=0)
        peak[peak <= 0] = np.nan
        writemom(peak, type='peak', filename=pth+basename, hdr=hd2d)
        vpeak = dil_mskcub.argmax_world(axis=0).to(u.km/u.s)
        writemom(vpeak, type='vpeak', filename=pth+basename, hdr=hd2d)
        vwidth = dil_mskcub_mom0 / (peak*np.sqrt(2*np.pi))
        vwidth[dil_mskcub_mom0 <= 0] = np.nan
        writemom(vwidth, type='vwidth', filename=pth+basename, hdr=hd2d)
    # --- Moment 1: mean velocity must be in range of cube
    dil_mskcub_mom1 = dil_mskcub.moment(order=1).to(u.km/u.s)
    vmin = dil_mskcub.spectral_extrema[0]
    vmax = dil_mskcub.spectral_extrema[1]
    dil_mskcub_mom1[dil_mskcub_mom1 < vmin] = np.nan
    dil_mskcub_mom1[dil_mskcub_mom1 > vmax] = np.nan
    dil_mskcub_mom1[nchanimg < mom1_minch] = np.nan
    writemom(dil_mskcub_mom1, type='mom1', filename=pth+basename, hdr=hd2d)
    # --- Moment 2: require at least 2 unmasked channels at each pixel
    dil_mskcub_mom2 = dil_mskcub.linewidth_sigma().to(u.km/u.s)
    dil_mskcub_mom2[nchanimg < mom2_minch] = np.nan
    writemom(dil_mskcub_mom2, type='mom2', filename=pth+basename, hdr=hd2d)
    #
    # --- CALCULATE ERRORS IN MOMENTS
    #
    altmom, errmom = calc_moments(image_cube, rms_cube, dilatedmask)
    errmom0 = errmom[0] * abs(hd3d['CDELT3'])/1000 * dil_mskcub_mom0.unit
    errmom0[nchanimg == 0] = np.nan
    writemom(errmom0, type='emom0', filename=pth+basename, hdr=hd2d)
    errmom1 = errmom[1] * u.km/u.s
    errmom1[dil_mskcub_mom1 < vmin] = np.nan
    errmom1[dil_mskcub_mom1 > vmax] = np.nan
    errmom1[nchanimg < mom1_minch] = np.nan
    writemom(errmom1, type='emom1', filename=pth+basename, hdr=hd2d)
    errmom2 = errmom[2] * u.km/u.s
    errmom2[nchanimg < mom2_minch] = np.nan
    writemom(errmom2, type='emom2', filename=pth+basename, hdr=hd2d)
    if altoutput:
        altmom0 = altmom[0] * abs(hd3d['CDELT3'])/1000 * dil_mskcub_mom0.unit 
        altmom0[nchanimg == 0] = np.nan
        altmom1 = altmom[1] * u.km/u.s
        altmom1[altmom1 < vmin] = np.nan
        altmom1[altmom1 > vmax] = np.nan
        altmom1[nchanimg < mom1_minch] = np.nan
        altmom2 = altmom[2] * u.km/u.s
        altmom2[nchanimg < mom2_minch] = np.nan
        writemom(altmom0, type='amom0', filename=pth+basename, hdr=hd2d)
        writemom(altmom1, type='amom1', filename=pth+basename, hdr=hd2d)
        writemom(altmom2, type='amom2', filename=pth+basename, hdr=hd2d)
    return

