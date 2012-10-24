# -*- coding: utf-8 -*-

import numpy as np
import cv2
FFT_MOD_FFTW = "fftw3"
FFT_MOD_ANFFT = "anfft"
try:
    import anfft
    USE_FFTW = True
    FFT_MODULE = FFT_MOD_ANFFT
    print "Found anfft"
except ImportError:
    print """Did not find module: anfft, try "pip install anfft" """
    print "Attempting to use python-fftw"
    try:
        import fftw3
        USE_FFTW = True
        FFT_MODULE = FFT_MOD_FFTW
        print "Found python-fftw"
    except ImportError:
        USE_FFTW = False
        print "Did not find module: fftw3"
        print "Using fallback numpy fft module..."
        print """Try "pip install anfft" or "apt-get install python-fftw" """
                

FLOAT_LIMITS = np.finfo(np.float)
DEBUG = True

class STRUCT(object):
    pass

class LogPolar(object):
    def __init__(self, n_rows, n_cols, wdim, rdim, tdim, threshold=600):
        #f1dim -> n_cols
        #f2_dim -> n_rows
        wdim = min([n_cols/2., n_rows/2., wdim])
        self.dims = STRUCT()
        self.dims.n_cols = n_cols
        self.dims.n_rows = n_rows
        self.dims.width = wdim
        self.dims.rho = rdim
        self.dims.theta = tdim
        self.dims.w1lcmap = 0
        self.dims.w2lcmap = 0
        
        self.vars = STRUCT()
        self.vars.thresh = threshold
        self.vars.base = 0
        self.vars.hscale = 0
        self.vars.vscale = 0
        
        self._index_ = STRUCT()
        self._index_.w_1 = np.empty((0, 0))
        self._index_.w_2 = np.empty((0, 0))
        self._create_map_()
    
    def _create_map_(self):
        # Base used to calculate log of radius
        # example: base^128 = 256
        # maps 256 rows in cartesian spectrum to 128 rows in polar plane
        # Maps n_rows rows into tdim rows
        self.vars.base = np.exp(np.log(self.dims.n_rows)/self.dims.theta)
        self.vars.hscale = ((self.dims.rho+self.vars.thresh)/
            (np.log2(self.dims.width-1)/np.log2(self.vars.base)))
        
        # Scaling to map 180 degrees in tdim samples
        self.vars.vscale = self.dims.theta/180.0
        
        # Min radius mapped into log-polar coordinates
        #rmin = base**((thresh+1)/hscale)
        
        ii = np.arange(self.dims.theta, dtype=np.float)
        jj = np.arange(self.dims.rho, dtype=np.float)
        # Vector of sampled angles
        theta = ii/self.vars.vscale
        # Vector of sampled radii
        rho = np.power(self.vars.base, 
                       (jj+self.vars.thresh)/self.vars.hscale)[np.newaxis]
        
        # (w1,w2) correspond to the cartesian coordinates in the freq plane
        theta_rad = np.deg2rad(theta)[:, np.newaxis]
        w_1 = np.dot(np.cos(theta_rad), rho)
        w_2 = np.dot(np.sin(theta_rad), rho)
        
        # Find nearest sample relative to center of image
        lcmap_j = np.round(w_1)
        lcmap_i = np.round(w_2)
        
        # Map image center relative indexing to standard array indexing
        self._index_.w_2 = (self.dims.n_rows/2-lcmap_i).astype(np.uint16)
        self._index_.w_1 = (self.dims.n_cols/2+lcmap_j).astype(np.uint16)
        self.dims.w2lcmap, self.dims.w1lcmap = lcmap_i.shape
        
    def transform(self, image):
        if DEBUG:
            assert self.isvalidimage(image), "Image dimensions don't match"
        # lp_im = np.zeros((self.dims.w2lcmap, self.dims.w1lcmap))
        # lp_im[self._index_.w_1, self._index_.w_2] = (
        #     image[self._index_.w_1, self._index_.w_2])
        return image[self._index_.w_2, self._index_.w_1]
        
    def isvalidimage(self, image):
        return image.shape == (self.dims.w2lcmap, self.dims.w1lcmap)
        

class PhaseCorrelator(object):
    def __init__(self, template=None, scene_dimensions=(1024, 768)):
        # Storage for the raw images
        self.images = STRUCT()
        self.images.raw = STRUCT()
        self.images.raw.template = None
        self.images.raw.scene = None
        self.images.raw.filter = None
        self.images.raw.window = np.zeros((0, 0))
        # Images padded to the same dimensions
        self.images.padded = STRUCT()
        self.images.padded.template = None
        self.images.padded.scene = None
        self.images.padded.filter = None
        self.images.padded.boxfilter = [None, None]
        # filtered versions of the padded images (obtained from ifft)
        self.images.filtered = STRUCT()
        self.images.filtered.template = None
        self.images.filtered.scene = None
        # logpolar images
        self.images.logpolar = STRUCT()
        self.images.logpolar.template = None
        self.images.logpolar.scene = None
        
        # DFTs of the padded images
        self.dft = STRUCT()
        self.dft.padded = STRUCT()
        self.dft.padded.template = None
        self.dft.padded.scene = None
        self.dft.padded.filter = None
        # DFTs of the filtered and padded images
        self.dft.filtered = STRUCT()
        self.dft.filtered.template = None
        self.dft.filtered.scene = None
        # Magnitude of the DFTs
        self.dft.magnitude = STRUCT()
        self.dft.magnitude.template = None
        self.dft.magnitude.scene = None
        self.dft.magnitude.filter = None
        # DFT of the log-polar images
        self.dft.logpolar = STRUCT()
        self.dft.logpolar.template = None
        self.dft.logpolar.scene = None
        
        # Scale and rotation estimate
        self.scalerot = STRUCT()
        self.scalerot.row = 0
        self.scalerot.col = 0
        self.scalerot.peak = 0
        self.scalerot.goodness = 0
        self.scalerot.scale = 1.0
        self.scalerot.rot = 0
        
        # Translation of template
        self.translation = STRUCT()
        self.translation.row = 0
        self.translation.col = 0
        self.translation.peak = 0
        self.translation.goodness = 0
        self.translation.x = 0
        self.translation.y = 0
        # This class instance is only compatible with images which can be 
        # padded to this size
        max_dim = np.max((scene_dimensions, template.shape))
        self.max_im_size = getOptimalDFTSize(max_dim)
        self.max_shape = (self.max_im_size, self.max_im_size)
        
        # Prepare template:
        self._generate_filter_fft_()
        h_min = 0.03
        tol = 0.001
        try:
            wdim1 = np.where(np.logical_and(
                ((h_min-tol) < self.dft.magnitude.filter[0]), 
                (self.dft.magnitude.filter[0] < (h_min+tol))))[0][-1]
        except:
            print "Could not compute w1 - setting to num_rows/2"
            wdim1 = self.max_im_size/2
        try:
            wdim2 = np.where(np.logical_and(
                ((h_min-tol) < self.dft.magnitude.filter[:, 0]), 
                (self.dft.magnitude.filter[:, 0] < (h_min+tol))))[0][-1]
        except:
            print "Could not compute w2 - setting to w1"
            wdim2 = wdim1
        # Log polar convertor
        rho_dim = self.max_im_size
        theta_dim = self.max_im_size
        width_dim = min([wdim1, wdim2])
        self.lpc = LogPolar(self.max_im_size, self.max_im_size, width_dim, 
                            rho_dim, theta_dim)
        self._create_fft_wisdom_
        self.set_template(template)
    
    def _create_fft_wisdom_(self):
        if USE_FFTW:
            print "Creating fft wisdom"
            zero_mat = np.zeros(self.max_shape)
            zero_mat_2 = np.zeros((self.max_shape[0], self.max_shape[1]*2))
            fft_zero = fft2(zero_mat, flags=['measure'])
            fft_zero_2 = fft2(zero_mat_2, flags=['measure'])
            ifft2(fft_zero, flags=['measure'])
            ifft2(fft_zero_2, flage=['measure'])
            
            
    def _generate_filter_fft_(self, kernel_size=41, sigma=1.5):
        # Prepare the LoG filter
        self.images.raw.filter = log_filter_kernel(kernel_size, sigma)
        self.images.padded.filter = symmetric_pad(self.images.raw.filter, 
                                                  self.max_shape)
        self.dft.padded.filter = fft2(self.images.padded.filter)
        n_rows, n_cols = self.max_shape
        k1 = np.arange(n_rows)[np.newaxis].T
        k2 = np.arange(n_cols)[np.newaxis]
        phase_correction = (
            np.dot(np.exp(1j*k1*2*np.pi/n_rows*(kernel_size-1)/2), 
                   np.exp(1j*k2*2*np.pi/n_cols*(kernel_size-1)/2)))
        self.dft.padded.filter *= phase_correction
        self.dft.magnitude.filter = np.abs(self.dft.padded.filter)
        
    def _prepare_image_(self, image):
        # Subtract the mean and multiply the image by a window to reduce 
        # edge effects
        if not (image.shape == self.images.raw.window.shape):
            self.images.raw.window = hann_window(image.shape)
        image = (image-image.mean())*self.images.raw.window
        # Pad the image to the correct size
        padded_image = symmetric_pad(image, self.max_shape)
        # Compute the DFT of the padded image
        padded_image_dft = fft2(padded_image)
        # Evaluate the filtered DFT
        filtered_padded_image_dft = padded_image_dft*self.dft.padded.filter
        # Magnitude of the filtered DFT
        filtered_padded_image_dft_mag = (
            np.fft.fftshift(np.abs(filtered_padded_image_dft)))
        # Log polar transform of the filtered DFT magnitude
        logpolar_image = (
            self.lpc.transform(filtered_padded_image_dft_mag))
        logpolar_dft_shape = (self.max_im_size, self.max_im_size*2)
        logpolar_dft = fft2(logpolar_image, 
                            logpolar_dft_shape)
        return (padded_image, padded_image_dft, filtered_padded_image_dft,
                filtered_padded_image_dft_mag, logpolar_image, logpolar_dft)
    
    def _prepare_template_(self):
        (self.images.padded.template, self.dft.padded.template, 
        self.dft.filtered.template, self.dft.magnitude.template,
        self.images.logpolar.template, self.dft.logpolar.template) = (
            self._prepare_image_(self.images.raw.template))
        # Compute the filtered image from its DFT
        self.images.filtered.template = (
            np.fft.fftshift(ifft2(self.dft.filtered.template).real))
    
    def prepare_scene(self, scene):
        self.images.raw.scene = scene
        (self.images.padded.scene, self.dft.padded.scene, 
        self.dft.filtered.scene, self.dft.magnitude.scene,
        self.images.logpolar.scene, self.dft.logpolar.scene) = (
            self._prepare_image_(self.images.raw.scene))
    
    def set_template(self, template_image):
        if not template_image is None:
            if DEBUG:
                assert (
                np.all(np.array(template_image.shape) < self.max_im_size)), (
                "Image size must be <= "+str(self.max_shape))
            self.images.raw.template = template_image.copy()
            self._prepare_template_()
    
    def _get_scale_rotation_(self):
        (row_val, col_val, peak_val, goodness) = (
            do_phase_correlate(self.dft.logpolar.scene,
                               self.dft.logpolar.template))
        scalefac = self.lpc.vars.base**(col_val/self.lpc.vars.hscale)
        rotation = -row_val/self.lpc.vars.vscale
        self.scalerot.row = row_val
        self.scalerot.col = col_val
        self.scalerot.peak = peak_val
        self.scalerot.goodness = goodness
        self.scalerot.scale = scalefac
        self.scalerot.rot = rotation
        print str((scalefac, 0))
    
    def _get_translation_(self):
        scalefac = self.scalerot.scale
        # Get new size of the template
        newsize = (
            np.array(self.images.filtered.template.shape)/scalefac).astype(int)
        # Check if the new size is valid
        
        if np.min(newsize) > self.max_im_size:
            row_val = col_val = peak_val = goodness = 0
            translation_x = translation_y = 0
        
        else:
            # Scale template - assume rotation is not 0
            scaled_template = cv2.resize(self.images.filtered.template, 
                                         tuple(newsize)[::-1], interpolation=2)
            # Pad the template to max size and compute the dft
            padded_scaled_template = symmetric_pad(scaled_template, 
                                                   self.max_shape)
            scaled_template_dft = fft2(padded_scaled_template)
            (row_val, col_val, peak_val, goodness) = (
                do_phase_correlate(self.dft.padded.scene, scaled_template_dft))
            translation_x = row_val
            translation_y = col_val
        
        self.translation.row = row_val
        self.translation.col = col_val
        self.translation.peak = peak_val
        self.translation.goodness = goodness
        self.translation.x = translation_x
        self.translation.y = translation_y
        print str((translation_x, translation_y))
    
    def estimate_scale_rot_trans(self, scene):
        # Prepare the scene
        self.prepare_scene(scene)
        # Determine the scaling and rotation
        self._get_scale_rotation_()
        # Estimate the translation
        self._get_translation_()
        
    def get_scale_rot_trans(self):
        return (self.scalerot.scale, self.scalerot.rot, 
                self.translation.x, self.translation.y)


def do_phase_correlate(fft_im1, fft_im2, *args):
    num_rows, num_cols = fft_im1.shape
    # Multiply the FFTs
    prod_fft = fft_im1*np.conj(fft_im2)
    # Compute the magnitude
    abs_prod_fft = np.abs(prod_fft) + FLOAT_LIMITS.tiny
    # Normalise
    cross_power_spectrum = prod_fft/abs_prod_fft
    # Obtain the correlation as the inverse FFT
    phase_corr_mat = np.fft.fftshift(ifft2(cross_power_spectrum).real)
    phase_corr_mat_sm = (
        cv2.boxFilter(phase_corr_mat, -1, (3, 3), normalize=True))
    (peak_row, peak_col) = np.unravel_index(phase_corr_mat_sm.argmax(), 
                                            phase_corr_mat_sm.shape)
    peak_val = phase_corr_mat_sm[peak_row, peak_col]
    goodness = (peak_val-phase_corr_mat_sm.mean())/phase_corr_mat_sm.std()
    row_val, col_val = (np.array([peak_row, peak_col])-
                        np.array([num_rows, num_cols])/2.0)
    return row_val, col_val, peak_val, goodness
    
def _fix_rfft_matrix_(dft, pad_shape):
    pad_cols = pad_shape[1] - dft.shape[1]
    dft_filled = np.zeros(pad_shape, dtype=np.complex)
    dft_filled[:, :dft.shape[1]] = dft
    dft_conj = np.flipud(np.fliplr(np.conj(dft[:, 1:pad_cols+1])))
    dft_filled[0, dft.shape[1]:] = dft_conj[-1, :]
    dft_filled[1:, dft.shape[1]:] = dft_conj[0:-1, :]
    return dft_filled

def symmetric_pad(arr, newshape, padval=0):
    num_rows, num_cols = newshape
    row_diff = num_rows-arr.shape[0]
    col_diff = num_cols-arr.shape[1]
    if DEBUG:
        assert ((row_diff>=0) and (col_diff>=0)), "New shape must be larger"
    top = np.int(np.ceil(row_diff/2.0))
    bottom = np.int(np.floor(row_diff/2.0))
    left = np.int(np.ceil(col_diff/2.0))
    right = np.int(np.floor(col_diff/2.0))
    return cv2.copyMakeBorder(arr, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=padval)


def hann_window(shape):
    return cv2.createHanningWindow(shape, cv2.CV_64F).T
    

def log_filter_kernel(kern_size=41, sigma=1.5):
    """
    Laplacian of Gaussian filter
    log_filter(kern_size=41, sigma=1.5) -> filter_kernel
    """
    # increment kern_size to next odd number
    kern_size += not bool(np.mod(kern_size, 2))
    kern_size = np.int(kern_size)
    xy_range = np.arange(-((kern_size-1)/2), ((kern_size+1)/2), 1)
    xy_sq_range = xy_range**2
    x_sq_vals, y_sq_vals = np.meshgrid(xy_sq_range, xy_sq_range)
    sigma_sq = sigma**2
    xsq_ysq_sum = x_sq_vals + y_sq_vals
    eps = np.finfo(np.float).eps
    
    # Gaussian kernel
    gaussian_kernel = np.exp(-(xsq_ysq_sum)/(2*sigma_sq))
    gaussian_kernel[gaussian_kernel < eps*gaussian_kernel.max()] = 0
    normalisation_factor = gaussian_kernel.sum() + eps
    gaussian_kernel /= normalisation_factor
    
    # Laplacian of Gaussian
    log_kernel = gaussian_kernel*(xsq_ysq_sum - 2*sigma_sq)/(sigma_sq**2)
    # make the filter sum to zero
    log_kernel -= log_kernel.sum()/(kern_size**2)
    return log_kernel
    

def getOptimalDFTSize(length):
    return cv2.getOptimalDFTSize(length)
    #return 2**np.ceil(np.log2(length)).astype(np.int)


def wfftn(arr, fft_shape=None, nthreads=4, flags=['estimate']):
    fft_shape = arr.shape if fft_shape is None else fft_shape
    if not arr.shape == fft_shape:
        pad_array = symmetric_pad(arr, fft_shape).astype(np.complex)
    else:
        pad_array = arr.astype(np.complex)
    outarray = np.zeros(fft_shape, dtype=np.complex) #pad_array.astype(np.complex)
    fft_forward = fftw3.Plan(pad_array, outarray, direction='forward', 
                             flags=flags, nthreads=nthreads)
    fft_forward()
    return outarray

def wrfftn(arr, fft_shape=None, nthreads=4, flags=['estimate']):
    fft_shape = arr.shape if fft_shape is None else fft_shape
    if not arr.shape == fft_shape:
        pad_array = symmetric_pad(arr, fft_shape)
    else:
        pad_array = arr
    num_cols = np.floor(fft_shape[1]/2) + 1
    outarray = np.zeros((fft_shape[0], num_cols), dtype=np.complex) #pad_array.astype(np.complex)
    fft_forward = fftw3.Plan(pad_array, outarray, direction='forward', 
                             flags=flags, nthreads=nthreads)
    fft_forward()
    return _fix_rfft_matrix_(outarray, fft_shape)

def wifftn(arr, fft_shape=None, nthreads=4, flags=['estimate']):
    fft_shape = arr.shape if fft_shape is None else fft_shape
    outarray = np.zeros(fft_shape, dtype=np.complex)
    if not arr.shape == fft_shape:
        pad_array = symmetric_pad(arr, fft_shape)
    else:
        pad_array = arr.copy()
    fft_backward = fftw3.Plan(pad_array, outarray, direction='backward', 
                              flags=flags, nthreads=nthreads)
    fft_backward()
    return outarray
    
def nprfft2(arr, fft_shape=None):
    fft_shape = arr.shape if fft_shape is None else fft_shape
    if not arr.shape == fft_shape:
        pad_array = symmetric_pad(arr, fft_shape)
    else:
        pad_array = arr
    outarray = np.fft.fft2(pad_array)
    return outarray

def npirfft2(arr, fft_shape=None):
    return np.fft.ifft2(arr, fft_shape)

def anfftn(arr, fft_shape=None):
    fft_shape = arr.shape if fft_shape is None else fft_shape
    if not arr.shape == fft_shape:
        pad_array = symmetric_pad(arr, fft_shape)
    else:
        pad_array = arr
    outarray = anfft.fftn(pad_array)
    return outarray

def anifftn(arr, *args, **kwargs):
    return anfft.ifftn(arr)

def anrfftn(arr, fft_shape=None):
    fft_shape = arr.shape if fft_shape is None else fft_shape
    if not arr.shape == fft_shape:
        pad_array = symmetric_pad(arr, fft_shape)
    else:
        pad_array = arr
    outarray = anfft.rfftn(pad_array)
    return _fix_rfft_matrix_(outarray, fft_shape)

def anirfftn(arr, *args, **kwargs):
    return anfft.irfftn(arr)
    

if USE_FFTW and FFT_MODULE in (FFT_MOD_FFTW, FFT_MOD_ANFFT):
    if FFT_MODULE == FFT_MOD_ANFFT:
        fft2 = anrfftn
        ifft2 = anifftn
    else:
        fft2 = wrfftn
        ifft2 = wifftn
else:
    fft2 = nprfft2
    ifft2 = npirfft2


def profile():
    tpl = cv2.imread("uwsim_panel_template.png", 0)
    newsize = tuple((np.array(tpl.shape)/3).astype(np.int))
    tpl2 = cv2.resize(tpl, newsize[::-1], interpolation=2)
    scn = cv2.imread("uwsim_camera_view_far.png", 0)
    panel_detector = PhaseCorrelator(tpl2.astype(np.float32), (1024, 768))
    empty = [panel_detector.estimate_scale_rot_trans(scn.astype(np.float32)) 
        for i in range(50)]
    print str(panel_detector.get_scale_rot_trans())

if __name__ == '__main__':
    profile()
