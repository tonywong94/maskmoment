# maskmoment
Masked moments of radio data cubes, using a dilated masking approach.  The basic idea is that the mask is produced from the cube itself, by taking a high significance contour (e.g. 4-sigma) and expanding it to a surrounding lower-significance contour (e.g. 2-sigma).  User has the option to smooth the data (spatially and/or spectrally) before defining these contours, and to pad the mask by a certain number of pixels in all directions.  An estimate of the noise at each location in the cube is needed, which can be provided from a different input cube, or can be calculated by the script from the data, optionally using a gain image.

Based on idl_mommaps: https://github.com/tonywong94/idl_mommaps

Required packages: [spectral_cube](https://spectral-cube.readthedocs.io/), [radio_beam](https://radio-beam.readthedocs.io/), [astropy](https://www.astropy.org), [scipy](https://www.scipy.org)

**Note:** A serious error was uncovered on 9-Jun-2020 where the emom0 image was not being scaled to K, even though it was labeled as K km/s.  Please update any results obtained with code from before that time.

Currently the code should **not** be installed in your site-packages area.  Just import
it using

    import sys
    sys.path.append('/path/to/maskmoment/')  
    from maskmoment import maskmoment

and then call it using

    maskmoment(img_fits, *other parameters*)

### Main Parameters (see code for additional options)

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
    mask_fits : FITS file name, optional
        External mask cube to use.  This cube should have 1's for valid pixels 
        and 0's for excluded pixels.  If this is provided then the mask generation
        is skipped and the program goes straight to calculating the moments.
    outdir : string, optional
        Directory to write the output files.
        Default: Write to the directory where img_fits resides.
        NOTE: Currently this directory is assumed to exist.
    outname : string, optional
        Basename for output files.  For instance, outname='foo' produces files
        'foo.mom0.fits.gz', etc.
        Default: Based on root name of img_fits.
    snr_hi : float, optional
        The high significance sigma threshold from which to begin mask dilation.
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
        If given as astropy quantity, should be given in velocity units.  
        If not given as astropy quantity, interpreted as number of channels.
        Default: No spectral smoothing is applied.
    vsm_type : string, optional
        What type of spectral smoothing to employ.  Currently three options:
        (1) 'boxcar' - 1D boxcar smoothing, vsm rounded to integer # of chans.
        (2) 'gauss' - 1D gaussian smoothing, vsm is the convolving gaussian FWHM.
        (3) 'gaussfinal' - 1D gaussian smoothing, vsm is the gaussian FWHM
            after convolution, assuming FWHM before convolution is 1 channel.        
        Default: 'gauss'
    perpixel : boolean, optional
        Whether to calculate the rms per XY pixel instead of over whole image.
        Set to True if you know there is a sensitivity variation across the image
        but you don't have a gain cube - requires rms_fits and gain_fits unset.
        Default: False
    output_snr_cube : boolean, optional
        Output the cube in SNR units in addition to the moment maps.
        Default: False
    output_snr_peak : boolean, optional
        Output the peak SNR image in addition to the moment maps.
        Default: False
    output_2d_mask : boolean, optional
        Output the projected 2-D mask as well as the newly generated mask.
        The projected mask at a given pixel is valid for all channels as
        long as the parent mask is valid for any channel.
        Default: False
    to_kelvin : boolean, optional
        Output the moment maps in K units if the cube is in Jy/beam units.
        Default: True

### Credits

Base code was developed by Tony Wong in 2019-2020.  Hailin Wang assisted with scripting, testing, and debugging the code during development.
