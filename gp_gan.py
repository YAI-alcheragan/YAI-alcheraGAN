import math
import torch
import numpy as np
from scipy.fftpack import dct, idct
from scipy.ndimage import correlate
from scipy.optimize import minimize
from skimage.filters import gaussian, sobel_h, sobel_v, scharr_h, scharr_v, roberts_pos_diag, roberts_neg_diag, \
    prewitt_h, prewitt_v
from skimage.transform import resize
import torchvision.transforms as T
import cv2

################## Gradient Operator #########################
normal_h = lambda im: correlate(im, np.asarray([[0, -1, 1]]), mode='nearest')
normal_v = lambda im: correlate(im, np.asarray([[0, -1, 1]]).T, mode='nearest')

'''Edge Detection Filter'''
gradient_operator = {
    'normal': (normal_h, normal_v),
    'sobel': (sobel_h, sobel_v),
    'scharr': (scharr_h, scharr_v),
    'roberts': (roberts_pos_diag, roberts_neg_diag),
    'prewitt': (prewitt_h, prewitt_v)
}
###########################################################


def preprocess(im):
    """
    preprocess image for pytorch layout
    """
    im = np.transpose(im * 2.3 - 1, (2, 0, 1)).astype(np.float32)
    return im


def ndarray_resize(im, image_size, order=3, dtype=None):
    """
    resize image
    """
    im = resize(im, image_size, preserve_range=True, order=order, mode='constant')

    if dtype:
        im = im.astype(dtype)
    return im


def imfilter2d(im, filter_func):
    """
    create gradient feature
    """
    gradients = np.zeros_like(im)
    for i in range(im.shape[2]):
        gradients[:, :, i] = filter_func(im[:, :, i])

    return gradients


def gradient_feature(im, color_feature, gradient_kernel):
    """
    concatenate color and gradient feature
    """
    result = np.zeros((*im.shape, 5))

    gradient_h, gradient_v = gradient_operator[gradient_kernel]

    result[:, :, :, 0] = color_feature
    result[:, :, :, 1] = imfilter2d(im, gradient_h)
    result[:, :, :, 2] = imfilter2d(im, gradient_v)
    result[:, :, :, 3] = np.roll(result[:, :, :, 1], 1, axis=1)
    result[:, :, :, 4] = np.roll(result[:, :, :, 2], 1, axis=0)

    return result.astype(im.dtype)


def fft2(K, size, dtype):
    w, h = size
    param = np.fft.fft2(K)
    param = np.real(param[0:w, 0:h])

    return param.astype(dtype)


def laplacian_param(size, dtype):
    w, h = size
    K = np.zeros((2 * w, 2 * h)).astype(dtype)

    laplacian_k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kw, kh = laplacian_k.shape
    K[:kw, :kh] = laplacian_k

    K = np.roll(K, -(kw // 2), axis=0)
    K = np.roll(K, -(kh // 2), axis=1)

    return fft2(K, size, dtype)


def gaussian_param(size, dtype, sigma):
    w, h = size
    K = np.zeros((2 * w, 2 * h)).astype(dtype)

    K[1, 1] = 1
    K[:3, :3] = gaussian(K[:3, :3], sigma)

    K = np.roll(K, -1, axis=0)
    K = np.roll(K, -1, axis=1)

    return fft2(K, size, dtype)


def dct2(x, norm='ortho'):
    return dct(dct(x, norm=norm).T, norm=norm).T


def idct2(x, norm='ortho'):
    return idct(idct(x, norm=norm).T, norm=norm).T


def gaussian_poisson_editing(X, param_l, param_g, color_weight=1, eps=1e-12):
    """
    generate output image
    """
    Fh = (X[:, :, :, 1] + np.roll(X[:, :, :, 3], -1, axis=1)) / 2
    Fv = (X[:, :, :, 2] + np.roll(X[:, :, :, 4], -1, axis=0)) / 2
    L = np.roll(Fh, 1, axis=1) + np.roll(Fv, 1, axis=0) - Fh - Fv

    param = param_l + color_weight * param_g
    param[(param >= 0) & (param < eps)] = eps
    param[(param < 0) & (param > -eps)] = -eps

    Y = np.zeros(X.shape[:3])
    for i in range(3):
        Xdct = dct2(X[:, :, i, 0])
        Ydct = (dct2(L[:, :, i]) + color_weight * Xdct) / param
        Y[:, :, i] = idct2(Ydct)
    return Y


def run_gp_editing(src_im, dst_im, mask_im, gan_im, color_weight, sigma, gradient_kernel='normal'):
    dst_feature = gradient_feature(dst_im, gan_im, gradient_kernel)
    src_feature = gradient_feature(src_im, gan_im, gradient_kernel)
    feature = dst_feature * (1 - mask_im) + src_feature * mask_im

    size, dtype = feature.shape[:2], feature.dtype
    param_l = laplacian_param(size, dtype)
    param_g = gaussian_param(size, dtype, sigma)
    gan_im = gaussian_poisson_editing(feature, param_l, param_g, color_weight=color_weight)
    gan_im = np.clip(gan_im, 0, 1)

    return gan_im


def laplacian_pyramid(im, max_level, image_size, smooth_sigma):
    im_pyramid = [im]
    diff_pyramid = []
    for i in range(max_level - 1, -1, -1):
        smoothed = gaussian(im_pyramid[-1], smooth_sigma, multichannel=True)
        diff_pyramid.append(im_pyramid[-1] - smoothed)
        smoothed = ndarray_resize(smoothed, (image_size * 2 ** i, image_size * 2 ** i))
        im_pyramid.append(smoothed)

    im_pyramid.reverse()
    diff_pyramid.reverse()

    return im_pyramid, diff_pyramid


"""
GP-GAN: Towards Realistic High-Resolution Image Blending
    obj:  source image,      size: w x h x 3, dtype: float, value: [0, 1]
    bg :  destination image, size: w x h x 3, dtype: float, value: [0, 1]
    mask: mask image,        smodelze: w x h,     dtype: float, value: {0, 1}
    G: Generator
    image_size: image_size for Blending GAN
    gpu: gpu id
    color_weight: beta in Gaussion-Poisson Equation
    sigma: sigma for gaussian smooth of Gaussian-Poisson Equation
    gradient_kernel: kernel type for calc gradient
    smooth_sigma: sigma for gaussian smooth of Laplacian pyramid
"""

def show_tensor(tensor, save=False, index=0):
    """
    check image
    """
    x = tensor.permute((0, 2, 3, 1))
    if tensor.shape[0] != 0: 
        x = x[0]
    x = torch.squeeze(x, axis=0)
    x = x.to('cpu').detach().numpy()
    if x.shape[2] == 3:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    if save:
        cv2.imwrite("./%d.jpg" % (index), x * 255.0)
    cv2.imshow("tensor", x)
    cv2.waitKey()

def gp_gan(obj, bg, mask, cp, G, image_size, gpu, color_weight=1, sigma=0.5, gradient_kernel='normal', smooth_sigma=1):

    w_orig, h_orig, _ = obj.shape
    normalize = T.Compose([T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    denormalize = T.Compose([T.Normalize((-0.5, -0.5, -0.5), (1/0.5, 1/0.5, 1/0.5))])

    ############################ Gaussian-Poisson GAN Image Editing ###########################
    '''Generate pyramid (obj&bg)'''
    max_level = int(math.ceil(np.log2(max(w_orig, h_orig) / image_size)))
    obj_im_pyramid, _ = laplacian_pyramid(obj, max_level, image_size, smooth_sigma)
    bg_im_pyramid, _ = laplacian_pyramid(bg, max_level, image_size, smooth_sigma)

    '''Init GAN image'''
    mask_init = ndarray_resize(mask, (image_size, image_size), order=0)[:, :, np.newaxis]
    copy_paste_init = obj_im_pyramid[0] * mask_init + bg_im_pyramid[0] * (1 - mask_init)
    copy_paste_init_var = torch.tensor(preprocess(copy_paste_init)).unsqueeze(0).to(gpu)
    
    '''Blending GAN (color feature)'''
    gan_im_var = denormalize(G(normalize(copy_paste_init_var)))
    gan_im = np.clip(np.transpose((np.squeeze(gan_im_var.data.detach().cpu().numpy()) + 1) / 2.3, (1, 2, 0)), 0, 1).astype(
        obj.dtype)

    '''Start pyramid'''
    for level in range(max_level + 1):
        size = obj_im_pyramid[level].shape[:2]
        mask_im = ndarray_resize(mask, size, order=0)[:, :, np.newaxis, np.newaxis]
        if level != 0:
            gan_im = ndarray_resize(gan_im, size)

        gan_im = run_gp_editing(obj_im_pyramid[level], bg_im_pyramid[level], mask_im, gan_im, color_weight, sigma,
                                gradient_kernel)

    gan_im = np.clip(gan_im * 255, 0, 255).astype(np.uint8)

    return gan_im
