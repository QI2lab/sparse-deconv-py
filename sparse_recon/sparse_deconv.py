import cupy as cp
import numpy as np
import warnings
import time

from matplotlib import pyplot as plt

from .sparse_hessian_recon.sparse_hessian_recon import sparse_hessian
from .iterative_deconv.iterative_deconv import iterative_deconv, alt_LR_decon
from .iterative_deconv.kernel import Gauss
from .utils.background_estimation import new_background_estimation
from .utils.upsample import spatial_upsample, fourier_upsample
try:
    import cupy as cp
except ImportError:
    cupy = None
xp = np if cp is None else cp
if xp is not cp:
    warnings.warn("could not import cupy... falling back to numpy & cpu.")
def sparse_deconv(img, sigma, sparse_iter = 100, fidelity = 150, sparsity = 10, tcontinuity = 0.5,
                          prior = 1, deconv_iter = 7, deconv_type = 1,
                          up_sample = 0,):

    """Sparse deconvolution.
   	----------
   	It is an universal post-processing framework for 
   	fluorescent (or intensity-based) image restoration, 
   	including xy (2D), xy-t (2D along t axis), 
   	and xy-z (3D) images. 
   	It is based on the natural priori 
   	knowledge of forward fluorescent 
   	imaging model: sparsity and 
   	continuity along xy-t(z) axes.
   	----------
    Parameters
    ----------
    img : ndarray
       Input image (can be N dimensional).
    sigma : 1/2/3 element(s) list
       The point spread function size in pixel.
    sparse_iter:  int, optional
         the iteration of sparse hessian {default: 100}
    fidelity : int, optional
       fidelity {default: 150}
    tcontinuity  : optional
       continuity along z-axial {default: 0.5}
    sparsity :  int, optional
        sparsity {default: 10}
    background:int, optional
        background estimation {default:1}:
        when background is none, 0
        when background is Weak background (HI), 1
        when background is Strong background (HI), 2
        when background is Weak background (LI), 3
        when background is with background (LI), 4
        when background is Strong background (LI), 5
    deconv_iter : int, optional
        the iteration of deconvolution {example:7}
    deconv_type : int, optional
       choose the different type deconvolution:
       0: No deconvolution
       1: LandWeber deconxolution
       2: Richardson-Lucy deconvolution

    Returns
    -------
    img_last : ndarray
       The sparse deconvolved image.

    Examples
    --------
    >>> from sparse_recon.sparse_deconv import sparse_deconv
	>>> from skimage import io
    >>> img = io.imread('test.tif')
	>>> img_recon = sparse_deconv(img, [5,5])
    References
    ----------
      [1] Weisong Zhao et al. Sparse deconvolution improves
      the resolution of live-cell super-resolution 
      fluorescence microscopy, Nature Biotechnology (2021),
      https://doi.org/10.1038/s41587-021-01092-2
    """
    if not sigma:
        print("The PSF's sigma is not given, turning off the iterative deconv...")
        deconv_type = 0
    img = img.astype(np.float32)
    scaler = np.max(img)
    img = img / scaler
    # remove background
    background, noise = new_background_estimation(img,prior=0,resolution_px=sigma, noise_lvl = 1)
    img = img - background
    img = img - noise
    img[img < 0] = 0.0
    img = img / (img.max())
    # up-sampling
    if up_sample == 1:
        img = fourier_upsample(img)
    elif up_sample == 2:
        img = spatial_upsample(img)
    img = img / (img.max())

    if up_sample == 1:
        img = fourier_upsample(img)
    elif up_sample == 2:
        img = spatial_upsample(img)
    img = img / (img.max())
    start = time.process_time()
    img_sparse = sparse_hessian(img, sparse_iter, fidelity, sparsity, tcontinuity)
    end = time.process_time()
    print('sparse hessian time')
    print(end - start)
    img_sparse = img_sparse / (img_sparse.max())

    if deconv_type == 0:
        img_last = img_sparse
        return scaler * img_last
    elif deconv_type == 3:
        start = time.process_time()
        sigma_3D = [.8/.230,sigma[0],sigma[1]]
        kernel_3D = Gauss(sigma_3D)
        img_decon = alt_LR_decon(img_sparse,kernel_3D,deconv_iter)
        end = time.process_time()
        print('deconv time')
        print(end - start)
        return scaler * img_decon
    else:
        start = time.process_time()
        sigma_3D = [.8/.230,sigma[0],sigma[1]]
        kernel = Gauss(sigma)
        img_last = iterative_deconv(img_sparse, kernel, deconv_iter, rule = deconv_type)
        end = time.process_time()
        print('deconv time')
        print(end - start)
        return scaler * img_last