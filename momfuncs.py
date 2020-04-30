import numpy as np
import os
from datetime import datetime
from scipy import ndimage
from spectral_cube import SpectralCube
from astropy.stats import mad_std
from astropy.io import fits
from astropy import units as u

import radio_beam
from astropy.convolution import Gaussian1DKernel
from astropy import wcs
from astropy.convolution import convolve, convolve_fft
from astropy.table import Table


def makenoise(cubearray, gainarray=None, rms=None, perpixel=False, edge=None, clip=0.2):
    """
    Given a data cube and an optional gain cube, estimate the noise at each position,
    using the robust mad_std estimator in astropy.

    Parameters
    ----------
    cubearray : SpectralCube or `~numpy.ndarray`
        The input data cube.  No default.
    gainarray : SpectralCube or `~numpy.ndarray`, optional
        Gain of a point source at each position of the cube (typically 0 to 1).  
        Can be 2-D or 3-D image, e.g. from pb.fits file from CASA.
        Default is to assume the rms is constant with position.
    rms : float, optional
        Global estimate of the rms noise, to be used instead of trying
        to estimate it from the data.  It should have the units of the input cube.
        Default is to calculate the rms from the data.
    perpixel : boolean, optional
        Whether to calculate the rms per pixel instead of over whole image.
        Set to True if you know there is a sensitivity variation across the image
        but you don't have a gain cube.  Only used if rms is unset.
        Default: False
    edge : int, optional
        Number of channels at left and right edges of each spectrum to use 
        for rms estimation.
        Default is to use all channels.
    clip : float, optional
        Pixels below this value in the input gainarray are marked invalid.
        Default: 0.2

    Returns
    -------
    noisearray : SpectralCube or `~numpy.ndarray`
        The noise estimate as a 3-D array with the same shape and units as 
        the input cube.
        If the input cube was a SpectralCube, a SpectralCube is returned.
    """
    if isinstance(cubearray, SpectralCube):
        hdr = cubearray.header
        spcube = True
        unit = cubearray.unit
        cubearray = cubearray.filled_data[:]
    else:
        spcube = False
    if perpixel==False:  # Calculate one rms for whole cube
        doax = None
    else:                # rms varies with position
        doax = 0
    if gainarray is None:
        if rms is None:
            if edge is None:
                rms = mad_std(cubearray, axis=doax, ignore_nan=True)
            else:  # take only edge channels for estimating rms
                slc = np.r_[0:edge,cubearray.shape[0]-edge:cubearray.shape[0]]
                rms = mad_std(cubearray[slc,:,:], axis=doax, ignore_nan=True)
        noisearray = np.zeros_like(cubearray) + rms
    else:
        if isinstance(gainarray, SpectralCube):
            gainarray = gainarray.unmasked_data[:]
        if clip is not None:
            gainarray[gainarray < clip] = np.nan
        if rms is None:
            imflat = cubearray * np.broadcast_to(gainarray, cubearray.shape)
            if edge is None:
                rms = mad_std(imflat, axis=doax, ignore_nan=True)
            else:  # take only edge channels for estimating rms
                slc = np.r_[0:edge,cubearray.shape[0]-edge:cubearray.shape[0]]
                rms = mad_std(imflat[slc,:,:], axis=doax, ignore_nan=True)
        noisearray = np.broadcast_to(rms/gainarray, cubearray.shape)
    print('Found rms value of', rms)
    if spcube:
        if unit != '':
            noisecube = SpectralCube(data=noisearray*unit, header=hdr, wcs=wcs.WCS(hdr))
        else:
            print('Noise cube has no units')
            noisecube = SpectralCube(data=noisearray, header=hdr, wcs=wcs.WCS(hdr))
        return noisecube
    else:
        return noisearray


# def snr_calc(image_cube, noise_cube):
#     """
#     Given a data cube and a noise cube, compute their ratio and its maximum
#     along the spectral dimension.
# 
#     Parameters
#     ----------
#     image_cube : SpectralCube
#         The input data cube.
#     noise_cube : SpectralCube or `~numpy.ndarray`
#         Noise estimate at each position of the cube.  Can be 2-D or 3-D image.
# 
#     Returns
#     -------
#     snr_cube : SpectralCube
#         The input data cube normalized by the noise.
#     snr_peak : `~numpy.ndarray`
#         The peak signal-to-noise ratio as a 2-D array.
#     """
#     header = image_cube.header
#     if isinstance(noise_cube, SpectralCube):
#         noise_cube = noise_cube.unitless_filled_data[:]
#     #snrarray = image_cube.unitless_filled_data[:] / noise_cube
#     snrarray = image_cube.filled_data[:] / noise_cube
#     snr_cube = SpectralCube(data=snrarray, header=header, wcs=wcs.WCS(header))
#     snr_peak = np.nanmax(snrarray, axis=0)
#     return snr_cube, snr_peak


def prunemask(maskarray, minarea=0, minch=2, byregion=True): 
    """
    Apply area and velocity width criteria on mask regions.

    Parameters
    ----------
    maskarray : `~numpy.ndarray`
        The 3-D mask array with 1s for valid pixels and 0s otherwise.
    minarea : int, optional
        Minimum velocity-integrated area in pixel units.
        Default: 0
    minch : int, optional
        Minimum velocity width in channel units.
        Default: 2
    byregion : boolean, optional
        Whether to enforce min velocity width on whole region, rather than by pixel
        Default: True

    Returns
    -------
    maskarray : `~numpy.ndarray`
        A copy of the input maskarray with regions failing the tests removed.
    """
    labeled_array_ch, num_features_ch = ndimage.label(maskarray)
    loc_objects = ndimage.find_objects(labeled_array_ch)
    for i in range(1, num_features_ch+1):
        loc = loc_objects[i-1]
        collapsereg = np.sum(maskarray[loc], axis=0)
        if np.count_nonzero(collapsereg) < minarea:
            maskarray[labeled_array_ch==i] = 0
        if byregion:
            if np.count_nonzero(collapsereg >= minch) == 0:
                maskarray[labeled_array_ch==i] = 0
        else:             
            flag = collapsereg < minch
            maskarray[loc][:,flag] = 0
    return maskarray      
        

def maskguard(maskarray, niter=1, xyonly=False, vonly=False): 
    """
    Pad a mask by specified number of pixels in all three dimensions.

    Parameters
    ----------
    maskarray : `~numpy.ndarray`
        The 3-D mask array with 1s for valid pixels and 0s otherwise.
    niter : int, optional
        Number of iterations for expanding mask by binary dilation.
        Default: 1
    xyonly : boolean, optional
        Whether to expand only in the two sky coordinates
        Default: False
    vonly : boolean, optional
        Whether to expand only in the spectral coordinate
        Default: False (ignored if xyonly==True)

    Returns
    -------
    maskarray : `~numpy.ndarray`
        A copy of the input maskarray after padding.
    """
    s = ndimage.generate_binary_structure(3, 1)
    if xyonly:
        s[0,:] = False
        s[2,:] = False
    elif vonly:
        s[1]=s[0]
    maskarray = ndimage.binary_dilation(maskarray, structure=s, iterations=niter)
    return maskarray


def dilmsk(snrcube, header=None, snr_hi=4, snr_lo=2, minbeam=1, min_thresh_ch=1, 
            min_tot_ch=2, nguard=[0,0], debug=False):
    """
    Dilate a mask from one specified threshold to another.

    Parameters
    ----------
    snrcube : SpectralCube or `~numpy.ndarray`
        The image cube normalized by the estimated RMS noise.  If a numpy array
        is provided the header must also be provided.
    header : `astropy.io.fits.Header`
        The cube FITS header, required if a numpy array is given.
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
    debug : boolean, optional
        Output a bunch of FITS files to diagnose what's going on.
        Default: False

    Returns
    -------
    dil_mask : `~numpy.ndarray`
        A binary mask array (0s and 1s) which can be applied to a cube
    """
    if isinstance(snrcube, SpectralCube):
        hdr = snrcube.header
    elif header is None:
        raise NameError('A header must be provided to dilmsk procedure')
    else:
        snrcube = SpectralCube(data=snrcube, header=header, wcs=wcs.WCS(header))
        hdr = header
    bmarea = ( 2*np.pi/(8*np.log(2)) * snrcube.beam.major.value * 
              snrcube.beam.minor.value / (hdr['cdelt2'])**2 )
    minarea = minbeam * bmarea
    print('Minimum area is {} pixels'.format(minarea))
    # High significance mask
    thresh_mask = (snrcube._data > snr_hi)
    # Low significance mask
    edge_mask = (snrcube._data > snr_lo)
    if debug:
        hdr['datamin'] = 0
        hdr['datamax'] = 1
        fits.writeto('snr_hi_mask.fits.gz', thresh_mask.astype(np.uint8), hdr, overwrite=True)
        fits.writeto('snr_lo_mask.fits.gz', edge_mask.astype(np.uint8), hdr, overwrite=True)
    # Require min_thresh_ch channels at all pixels in high significance mask
    if min_thresh_ch > 1:
        thresh_mask = prunemask(thresh_mask, minch=min_thresh_ch, byregion=False)
        if debug:
            fits.writeto('snr_hi_mask_minch.fits.gz', thresh_mask.astype(np.uint8), hdr, overwrite=True)
    # Find islands in low significance mask
    if snr_lo < snr_hi:
        s = ndimage.generate_binary_structure(3, 1)
        labeled_edge, num_edge = ndimage.label(edge_mask, structure=s)
        if debug:
            hdr['datamax'] = num_edge
            fits.writeto('labeled_edge.fits.gz', labeled_edge.astype(np.uint8), hdr, overwrite=True)
        print('Found {} objects with SNR above {}'.format(num_edge, snr_lo))
        # Reject islands which do not reach high significance threshold
        hieval = labeled_edge[thresh_mask]
        dil_mask = np.isin(labeled_edge, hieval) & edge_mask
        if debug:
            hdr['datamax'] = 1
            fits.writeto('snr_mrg_mask.fits.gz', dil_mask.astype(np.uint8), hdr, overwrite=True)
    else:
        dil_mask = thresh_mask
    # Final pruning to enforce area and total vel width constraints
    if min_tot_ch > 1 or minarea > 0:
        dil_mask = prunemask(dil_mask, minarea=minarea, minch=min_tot_ch)
        if debug:
            fits.writeto('pruned_mask.fits.gz', dil_mask.astype(np.uint8), hdr, overwrite=True)
    # Expand by nguard pixels (padding)
    if sum(nguard) > 0:
        if nguard[0] == 0:
            dil_mask = maskguard(dil_mask, niter=nguard[1], vonly=True)
        elif nguard[1] == 0:
            dil_mask = maskguard(dil_mask, niter=nguard[0], xyonly=True)
        else:
            if nguard[1] != nguard[0]:
                print('Warning: setting nguard = [{},{}]'.format(nguard[0],nguard[0]))
            dil_mask = maskguard(dil_mask, niter=nguard[0])
        if debug:
            fits.writeto('guardband_mask.fits.gz', dil_mask.astype(np.uint8), hdr, overwrite=True)
    return dil_mask



def smcube(snrcube, header=None, fwhm=10*u.arcsec, vsm=None, edgech=None):
    """
    Smooth an SNRcube to produce a higher signal-to-noise SNRcube.

    Parameters
    ----------
    snrcube : SpectralCube or `~numpy.ndarray`
        The image cube normalized by the estimated RMS noise.  If a numpy array
        is provided the header must also be provided.
    header : `astropy.io.fits.Header`
        The cube FITS header, required if a numpy array is given.
    fwhm : float or :class:`~astropy.units.Quantity`, optional
        Final spatial resolution to smooth to.  If not astropy quantity, assumed
        to be given in arcsec.
        Default: 10 arcsec
    vsm : float or :class:`~astropy.units.Quantity`, optional
        Final velocity resolution to smooth to.  If not astropy quantity, assumed
        to be given in km/s.
        Default: No spectral smoothing is applied.
    edgech : int, optional
        Number of channels at left and right edges of each spectrum to use 
        for rms estimation.
        Default is to use all channels.

    Returns
    -------
    sm_snrcube : SpectralCube
        A cube is SNR units after smoothing to the desired resolution.
    """
    if isinstance(snrcube, SpectralCube):
        hdr = snrcube.header
    elif header is None:
        raise NameError('A header must be provided to smcube procedure')
    else:
        snrcube = SpectralCube(data=snrcube, header=header, wcs=wcs.WCS(header))
        hdr = header
        print(snrcube)
    
    # -- Spatial smoothing
    # Requested final resolution
    if not hasattr(fwhm, 'unit'):
        fwhm = fwhm * u.arcsec
    sm_beam = radio_beam.Beam(major=fwhm, minor=fwhm, pa=0*u.deg)
    print('Convolving to', sm_beam)
    # From convolve_to method in spectral_cube
    pixscale = wcs.utils.proj_plane_pixel_area(snrcube.wcs.celestial)**0.5*u.deg
    if hasattr(snrcube, 'beam'):
        print('Existing', snrcube.beam)
        convolution_kernel = sm_beam.deconvolve(snrcube.beam).as_kernel(pixscale)
    else:
        print('Warning: no existing beam found in input to smcube')
        convolution_kernel = sm_beam.as_kernel(pixscale)
    sm_snrcube = snrcube.spatial_smooth(convolution_kernel, convolve_fft, 
                  fill_value=0.0, nan_treatment='fill', parallel=False)
#     sm_snrcube.write('sm_snrcube.fits.gz', overwrite=True)
    
    # -- Spectral smoothing
    if vsm is not None:
        current_resolution = hdr['CDELT3']/1000. * u.km/u.s
        if hasattr(vsm, 'unit'):
            target_resolution = vsm
        else:
            target_resolution = vsm * u.km/u.s
        fwhm_factor = np.sqrt(8*np.log(2))
        if target_resolution > current_resolution:
            gaussian_width = ((target_resolution**2 - current_resolution**2)**0.5 
                              / current_resolution / fwhm_factor).value
            print('Vel smooth kernel has stddev of {} channels'.format(gaussian_width))
            kernel = Gaussian1DKernel(gaussian_width)
            sm2_snrcube = sm_snrcube.spectral_smooth(kernel)
            sm_snrcube = sm2_snrcube
        else:
            print('ERROR: requested resolution of', target_resolution,
                  'is less than current resolution of', current_resolution)

    # -- Renormalize by rms
    newrms = makenoise(sm_snrcube, edge=edgech)
#     newrms.write('newrms.fits.gz', overwrite=True)
    sm_snrcube = sm_snrcube / newrms
#     sm_snrcube.write('sm_snrcube2.fits.gz', overwrite=True)
    return sm_snrcube


def findflux(imcube, rmscube, mask=None):
    """
    Calculate integrated spectrum and total integrated flux.

    Parameters
    ----------
    imcube : SpectralCube
        The image cube over which to measure the flux.
    rmscube : SpectralCube
        A cube representing the noise estimate at each location in the image
        cube.  Should have the same units as the image cube.
    mask : `~numpy.ndarray`
        A binary mask array (0s and 1s) to be applied before measuring the flux
        and uncertainty.  This should NOT be a SpectralCube.

    Returns
    -------
    fluxtab : :class:`~astropy.table.Table`
        A 3-column table providing the velocity, flux, and uncertainty.
    """
    
    hd = imcube.header
    CDELT1, CDELT2 = hd['CDELT1']*u.deg, hd['CDELT2']*u.deg
    # The beam area in pixels, used for conv to Jy and for correlated noise
    beamarea = (imcube.beam.sr).to(u.deg**2)/abs(CDELT1*CDELT2)
    vels = imcube.spectral_axis.to(u.km/u.s)
    delv = abs(vels[1]-vels[0])

    if mask is not None:
        immask   = imcube.with_mask(mask > 0)
        errmask  = rmscube.with_mask(mask > 0)
    else:
        immask = imcube
        errmask = rmscube
    specflux = np.nansum(immask.unitless_filled_data[:],axis=(1,2))
    totflux  = np.nansum(immask.unitless_filled_data[:]) * delv.value
    specvar  = np.nansum(errmask.unitless_filled_data[:]**2,axis=(1,2))
    totvar   = np.nansum(errmask.unitless_filled_data[:]**2)
    specerr  = np.sqrt(specvar*beamarea)
    toterr   = np.sqrt(totvar*beamarea) * delv.value

    if imcube.unit == 'Jy / beam':
        specflux = specflux/beamarea
        specerr  = specerr/beamarea
        totflux  = totflux/beamarea
        toterr   = toterr/beamarea
        out_unit = u.Jy
    else:
        out_unit = imcube.unit * u.pix

    fluxtab = Table([vels, specflux, specerr], names=('Velocity', 'Flux', 'FluxErr'))
    fluxtab.meta['totflux'] = "{:.2f} +/- {:.2f}".format(totflux,toterr)+' '+out_unit.to_string()+' km/s'
    fluxtab['Velocity'].description = 'Velocity, ' + hd['ctype3'] + ', ' + hd['specsys']
    fluxtab['Flux'].unit = out_unit
    fluxtab['Flux'].format = '.4f'
    fluxtab['Flux'].description = 'Flux after masking'
    fluxtab['FluxErr'].unit = out_unit
    fluxtab['FluxErr'].format = '.4f'
    fluxtab['FluxErr'].description = 'Formal uncertainty in masked flux'

    return fluxtab

# tb = image cube
# nse = noise cube
# vel = vchan broadcast to image cube
# 
# mom0  = np.nansum(tb, axis=0)
# mom0_var = np.nansum( nse**2, axis=0 )
# mom0_err = np.sqrt(mom0_var)
# 
# mom1  = np.nansum(tb*vel, axis=0) / mom0
# mom1_var = np.nansum( ((vel - mom1)/mom0 * nse)**2, axis=0 )
# mom1_err = np.sqrt(mom1_var)
# 
# mom2  = np.nansum(tb*(vel-mom1)**2, axis=0)/mom0
# mom2_var = np.nansum( ((mom0*(vel-mom1)**2-np.nansum(tb*(vel-mom1)**2, axis=0))/mom0**2 * nse)**2 + (2*np.nansum(tb*(vel-mom1), axis=0)/mom0 * mom1_err)**2, axis=0 )
# stdev = np.sqrt(mom2)
# sderr = np.sqrt(mom2_var)/(2*stdev)
