from pywt import wavedecn, waverecn
from scipy.ndimage import gaussian_filter  
from joblib import Parallel, delayed 
import multiprocessing
import numpy as np

# new wavelet decomposition from https://github.com/NienhausLabKIT/HuepfelM/tree/master/WBNS/python_script
#https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-12-2-969&id=446834

def wavelet_based_BG_subtraction(img,num_levels,noise_lvl):

  coeffs = wavedecn(img, 'db1', level=None) #decomposition
  coeffs2 = coeffs.copy()
  
  for BGlvl in range(1, num_levels):
      coeffs[-BGlvl] = {k: np.zeros_like(v) for k, v in coeffs[-BGlvl].items()} #set lvl 1 details  to zero
  
  Background = waverecn(coeffs, 'db1') #reconstruction
  del coeffs
  BG_unfiltered = Background
  Background = gaussian_filter(Background, sigma=2**num_levels) #gaussian filter sigma = 2^#lvls 
  
  coeffs2[0] = np.ones_like(coeffs2[0]) #set approx to one (constant)
  for lvl in range(1, np.size(coeffs2)-noise_lvl):
      coeffs2[lvl] = {k: np.zeros_like(v) for k, v in coeffs2[lvl].items()} #keep first detail lvl only
  Noise = waverecn(coeffs2, 'db1') #reconstruction
  del coeffs2
  
  return Background, Noise, BG_unfiltered

def new_background_estimation(img,prior=0,resolution_px=[5,5], noise_lvl = 1):

    if prior == 0:
        img = img
    elif prior == 1:
        img = img / 2.5
    elif prior == 2:
        img = img / 2.0
    elif prior == 3:
        mean = np.mean(img) / 2.5
        img[img > mean] = mean
    elif prior == 4:
        mean= np.mean(img)
        img[img > mean] = mean
    elif prior == 5:
        mean = np.mean(img) / 2
        img[img > mean] = mean

    #number of levels for background estimate
    num_levels = np.uint16(np.ceil(np.log2(resolution_px[0])))

    #image = np.array(io.imread(os.path.join(data_dir, file)),dtype = 'float32')
    if np.ndim(img) == 2:
        shape = np.shape(img)
        img = np.reshape(img, [1, shape[0], shape[1]])
    shape = np.shape(img)
    if shape[1] % 2 != 0:
        img = np.pad(img,((0,0), (0,1), (0, 0)), 'edge')
        pad_1 = True
    else:
        pad_1 = False
    if shape[2] % 2 != 0:
        img = np.pad(img,((0,0), (0,0), (0, 1)), 'edge')
        pad_2 = True
    else:
        pad_2 = False

    #extract background and noise
    num_cores = multiprocessing.cpu_count() #number of cores on your CPU
    res = Parallel(n_jobs=num_cores,max_nbytes=None)(delayed(wavelet_based_BG_subtraction)(img[slice],num_levels, noise_lvl) for slice in range(np.size(img,0)))
    background, noise, bg_unfiltered = zip(*res)

    #convert to float64 numpy array
    noise = np.asarray(noise,dtype = 'float32')
    background = np.asarray(background,dtype = 'float32')
    bg_unfiltered = np.asarray(bg_unfiltered,dtype = 'float32')

    #undo padding
    if pad_1:
        img = img[:,:-1,:]
        noise = noise[:,:-1,:]
        background = background[:,:-1,:]
        bg_unfiltered = bg_unfiltered[:,:-1,:]
    if pad_2:
        img = img[:,:,:-1]
        noise = noise[:,:,:-1]
        background = background[:,:,:-1]
        bg_unfiltered = bg_unfiltered[:,:,:-1]

    #correct noise
    noise[noise<0] = 0 #positivity constraint
    noise_threshold = np.mean(noise)+2*np.std(noise)
    noise[noise>noise_threshold] = noise_threshold #2 sigma threshold reduces artifacts

    return background, noise