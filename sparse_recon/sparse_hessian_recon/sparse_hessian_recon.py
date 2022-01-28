
import gc
from .operation import *
from .sparse_iteration import *
import dask.array as da

import numpy as np
try:
    import cupy as cp
except ImportError:
    cupy = None
xp = np if cp is None else cp
if xp is not cp:
    warnings.warn("could not import cupy... falling back to numpy & cpu.")

def sparse_hessian_dask(f, iteration_num = 100, fidelity = 150, sparsity = 10, contiz = 0.5 , mu = 1):

    # define chunk size for GPU processing
    chunk_size = (128,351,600)

    # calculate amount to pad array to be even multiple of chunk size
    arr_size_mismatch = np.divide(f.shape,chunk_size)
    factor_z = np.round((chunk_size[0]*np.ceil(arr_size_mismatch[0])) - f.shape[0],0).astype(int)
    factor_y = np.round((chunk_size[1]*np.ceil(arr_size_mismatch[1])) - f.shape[1],0).astype(int)
    factor_x = np.round((chunk_size[2]*np.ceil(arr_size_mismatch[2])) - f.shape[2],0).astype(int)

    # pad array
    f_padded = np.pad(f,((0,factor_z),(0,factor_y),(0,factor_x)),mode='constant',constant_values=np.median(f))
    del f
    gc.collect()

    # create dask array
    arr = da.from_array(f_padded, chunks=chunk_size)

    def sparse_hessian_chunk(chunk):
        chunked_result = sparse_hessian(chunk,iteration_num, fidelity, sparsity, contiz, mu)
        return chunked_result

    # run overlapped sparse hessian calculation
    result_overlap = arr.map_overlap(sparse_hessian_chunk,depth=(0,0,12), boundary='reflect', dtype='float32').compute(num_workers=1)

    return xp.asnumpy(result_overlap)

def sparse_hessian(f, iteration_num = 100, fidelity = 150, sparsity = 10, contiz = 0.5 , mu = 1):
    '''
    function g = SparseHessian_core(f,iteration_num,fidelity,sparsity,iteration,contiz,mu)
    -----------------------------------------------
    Source code for argmin_g { ||f-g ||_2^2 +||gxx||_1+||gxx||_1+||gyy||_1+lamdbaz*||gzz||_1+2*||gxy||_1
     +2*sqrt(lamdbaz)||gxz||_1+ 2*sqrt(lamdbaz)|||gyz||_1+2*sqrt(lamdbal1)|||g||_1}
     f : ndarray
       Input image (can be N dimensional).
     iteration_num:  int, optional
        the iteration of sparse hessian {default:100}
     fidelity : int, optional
       fidelity {default: 150}
     contiz  : int, optional
       continuity along z-axial {example:1}
     sparsity :  int, optional
        sparsity {example:15}
    ------------------------------------------------
    Output:
      g
    '''
    if xp is not cp:
        contiz = np.sqrt(contiz)
        f1 = f
    else:
        contiz = cp.sqrt(contiz)
        f1 = cp.asarray(f, dtype = 'float32')
    flage = 0
    # f = cp.divide(f,cp.max(f[:]))
    f_flag = f.ndim
    if f_flag == 2:
        contiz = 0
        flage = 1
        f = xp.zeros((3,f.shape[0], f.shape[1]), dtype = 'float32')
        f = xp.array(f)
        for i in range(0,3):
            f[i,:,:] = f1
        
    elif f_flag > 2:
        if f1.shape[0] < 3:
            contiz = 0
            f = xp.zeros((3, f.shape[1], f.shape[2]), dtype = 'float32')
            f[0:f1.shape[0],:,:] = f1
            for i in range(f1.shape[0], 3):
                f[i, :, :] = f[1,:,:]
        else:
             f = f1
    imgsize = xp.shape(f)

    print("Start the Sparse deconvolution...")
    ## calculate derivate
    xxfft = operation_xx(imgsize)
    yyfft = operation_yy(imgsize)
    zzfft = operation_zz(imgsize)
    xyfft = operation_xy(imgsize)
    xzfft = operation_xz(imgsize)
    yzfft = operation_yz(imgsize)

    operationfft = xxfft + yyfft + (contiz**2)*zzfft+ 2*xyfft +2*(contiz)*xzfft + 2*(contiz)*yzfft
    normlize = (fidelity/mu) + (sparsity**2) + operationfft
    del xxfft,yyfft,zzfft,xyfft,xzfft,yzfft,operationfft
    gc.collect()
    xp.clear_memo()
    ## initialize b
    bxx = xp.zeros(imgsize,dtype='float32')
    byy = bxx
    bzz = bxx
    bxy = bxx
    bxz = bxx
    byz = bxx
    bl1 = bxx
    ## initialize g
    g_update = xp.multiply(fidelity / mu, f)
    ## iteration
    for iter in range(0, iteration_num):

        g_update = xp.fft.fftn(g_update)

        if iter == 0:
            g = xp.fft.ifftn(g_update / (fidelity / mu)).real

        else:
            g = xp.fft.ifftn(xp.divide(g_update, normlize)).real

        g_update =xp.multiply((fidelity / mu), f)

        Lxx,bxx = iter_xx(g, bxx, 1, mu)
        g_update = g_update + Lxx
        del Lxx
        gc.collect()
        
        Lyy,byy = iter_yy(g, byy, 1, mu)
        g_update = g_update + Lyy
        del Lyy
        gc.collect()

        Lzz,bzz = iter_zz(g, bzz, contiz**2, mu)
        g_update = g_update + Lzz
        del Lzz
        gc.collect()

        Lxy,bxy = iter_xy(g, bxy, 2, mu)
        g_update = g_update + Lxy
        del Lxy
        gc.collect()

        Lxz,bxz = iter_xz(g, bxz, 2 * contiz, mu)
        g_update = g_update + Lxz
        del Lxz
        gc.collect()

        Lyz,byz = iter_yz(g, byz, 2 * contiz, mu)
        g_update = g_update + Lyz
        del Lyz
        gc.collect()

        Lsparse,bl1 = iter_sparse(g, bl1, sparsity, mu)
        g_update = g_update + Lsparse
        del Lsparse
        gc.collect()

        print('%d iterations done\r' % iter, end="")
    g[g < 0] = 0
    print("")


    del bxx,byy,bzz,bxy,byz,bl1,f,normlize,g_update
    gc.collect()

    return g[1, :, :] if flage else g
