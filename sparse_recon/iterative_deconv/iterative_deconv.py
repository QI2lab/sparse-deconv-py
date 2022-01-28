import math
import warnings
import numpy as np
from numpy import zeros
from cucim.skimage.restoration import richardson_lucy as lr_cucim
import gc
import dask.array as da

try:
    import cupy as cp
except ImportError:
    cupy = None

xp = np if cp is None else cp

if xp is not cp:
    warnings.warn("could not import cupy... falling back to numpy & cpu.")


def alt_LR_decon_dask(img,psf,iteration):

    # define chunk size for GPU processing
    chunk_size = (128,351,900)

    # calculate amount to pad array to be even multiple of chunk size
    arr_size_mismatch = np.divide(img.shape,chunk_size)
    factor_z = np.round((chunk_size[0]*np.ceil(arr_size_mismatch[0])) - img.shape[0],0).astype(int)
    factor_y = np.round((chunk_size[1]*np.ceil(arr_size_mismatch[1])) - img.shape[1],0).astype(int)
    factor_x = np.round((chunk_size[2]*np.ceil(arr_size_mismatch[2])) - img.shape[2],0).astype(int)

    # pad array
    img_padded = np.pad(img,((0,factor_z),(0,factor_y),(0,factor_x)),mode='constant',constant_values=np.median(img))
    del img
    gc.collect()

    # create dask array
    arr = da.from_array(img_padded, chunks=chunk_size)

    def sparse_alt_LR_chunk(chunk):
        chunked_result = alt_LR_decon(chunk,psf,iteration)
        return chunked_result

    # run overlapped sparse hessian calculation
    result_overlap = arr.map_overlap(sparse_alt_LR_chunk,depth=(0,0,psf.shape[2]), boundary='reflect', dtype='float32').compute(num_workers=1)

    return xp.asnumpy(result_overlap)

def alt_LR_decon(img,psf,iteration):
    img_float16 = xp.asarray(img.astype(np.float16))
    psf_float16 = xp.asarray(psf.astype(np.float16))
    img_decon = lr_cucim(img_float16, psf_float16, iterations=iteration, clip=True, filter_epsilon=1e-6)
    img_decon_np = xp.asnumpy(img_decon)
    
    del img_float16, psf_float16, img_decon
    gc.collect()
    xp.clear_memo()

    return img_decon_np

def iterative_deconv(data,kernel,iteration,rule):
    if xp is not np:
        data = xp.asarray(data)
        kernel = xp.asarray(kernel)

    if data.ndim > 2:
        data_de = xp.zeros((data.shape[0], data.shape[1],data.shape[2]), dtype = 'float32')
        for i in range(0, data.shape[0]):
            data_de[i, :, ] = (deblur_core(data[i, :,:], kernel, iteration, rule)).real
    else:
        data_de = (deblur_core(data, kernel, iteration, rule)).real

    if xp is not np:
        data_de = xp.asnumpy(data_de)

    return data_de

def deblur_core(data, kernel, iteration, rule):

    #data = cp.asnumpy(data)
    kernel = xp.array(kernel)
    kernel = kernel / sum(sum(kernel))
    kernel_initial = kernel
    [dx,dy] = data.shape

    B = math.floor(min(dx,dy)/6)
    data = xp.pad(data, [int(B),int(B)], 'edge')
    yk = data
    xk = zeros((data.shape[0], data.shape[1]), dtype = 'float32')
    vk = zeros((data.shape[0], data.shape[1]), dtype = 'float32')
    otf = psf2otf(kernel_initial, data.shape)

    if rule == 2: 
    #LandWeber deconv
        t = 1
        gamma1 = 1
        for i in range(0,iteration):

            if i == 0:
                xk_update = data

                xk = data + t*xp.fft.ifftn(xp.conj(otf)) * (xp.fft.fftn(data) - (otf *xp.fft.fftn(data)))
            else:
                gamma2 = 1/2*(4 * gamma1*gamma1 + gamma1**4)**(1/2) - gamma1**2
                beta = -gamma2 *(1 - 1 / gamma1)
                yk_update = xk + beta * (xk - xk_update)
                yk = yk_update + t * xp.fft.ifftn(xp.conj(otf) * (xp.fft.fftn(data) - (otf * xp.fft.fftn(yk_update))))
                yk = xp.maximum(yk, 1e-6, dtype = 'float32')
                gamma1 = gamma2
                xk_update = xk
                xk = yk

    elif rule == 1:
    #Richardson-Lucy deconv

        for iter in range(0, iteration):

            xk_update = xk
            rliter1 = rliter(yk, data, otf)

            xk = yk * ((xp.fft.ifftn(xp.conj(otf) * rliter1)).real) / ( (xp.fft.ifftn(xp.fft.fftn(xp.ones(data.shape)) * otf)).real)

            xk = xp.maximum(xk, 1e-6, dtype = 'float32')

            vk_update = vk

            vk =xp.maximum(xk - yk, 1e-6 , dtype = 'float32')

            if iter == 0:
                alpha = 0
                yk = xk
                yk = xp.maximum(yk, 1e-6,dtype = 'float32')
                yk = xp.array(yk)

            else:

                alpha = sum(sum(vk_update * vk))/(sum(sum(vk_update * vk_update)) + math.e)
                alpha = xp.maximum(xp.minimum(alpha, 1), 1e-6, dtype = 'float32')
               # start = time.process_time()
                yk = xk + alpha * (xk - xk_update)
                yk = xp.maximum(yk, 1e-6, dtype = 'float32')
                yk[xp.isnan(yk)] = 1e-6
                #end = time.process_time()
               # print(start, end)
                #K=np.isnan(yk)

    yk[yk < 0] = 0
    yk = xp.array(yk, dtype = 'float32')
    data_decon = yk[B + 0:yk.shape[0] - B, B + 0: yk.shape[1] - B]

    return data_decon

def cart2pol(x, y):
    rho = xp.sqrt(x ** 2 + y ** 2)
    phi = xp.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * xp.cos(phi)
    y = rho * xp.sin(phi)
    return (x, y)

def psf2otf(psf, outSize):
    psfSize = xp.array(psf.shape)
    outSize = xp.array(outSize)
    padSize = xp.array(outSize - psfSize)
    psf = xp.pad(psf, ((0, int(padSize[0])), (0, int(padSize[1]))), 'constant')
    for i in range(len(psfSize)):
        psf = xp.roll(psf, -int(psfSize[i] / 2), i)
    otf = xp.fft.fftn(psf)
    nElem = xp.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * xp.log2(psfSize[k]) * nffts
    if xp.max(xp.abs(xp.imag(otf))) / xp.max(xp.abs(otf)) <= nOps * xp.finfo(xp.float32).eps:
        otf = xp.real(otf)
    return otf

def rliter(yk,data,otf):
    rliter = xp.fft.fftn(data / xp.maximum(xp.fft.ifftn(otf * xp.fft.fftn(yk)), 1e-6))
    return rliter

