# maskmoment
Masked moments of radio data cubes, using a dilated masking approach.

Based on idl_mommaps: https://github.com/tonywong94/idl_mommaps

Required packages: [spectral_cube](https://spectral-cube.readthedocs.io/), [radio_beam](https://radio-beam.readthedocs.io/), [astropy](https://www.astropy.org), [scipy](https://www.scipy.org)

Currently this should **not** be installed in your site-packages area.  Just import
it using

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
    min_tot_all : boolean, optional
        Enforce min_tot_ch for all pixels instead of for regions as a whole.
        Default: False
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
    output_snr_cube : boolean, optional
        Output the cube in SNR units in addition to the moment maps.
        Default: False
    to_kelvin : boolean, optional
        Output the moment maps in K units if the cube is in Jy/beam units.
        Default: True
