# Use DRUNet denoiser

import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

#import denoisers.DPIR.utils.utils_logger as utils_logger
import denoisers.DPIR.utils.utils_model as utils_model
import denoisers.DPIR.utils.utils_image as util

#from utils import utils_logger
#from utils import utils_model
#from utils import utils_image as util


def my_Denoiser(image_in, sigma_denoiser):
# ----------------------------------------
# Preparation
# ----------------------------------------

    noise_level_img = sigma_denoiser                 # set AWGN noise level for noisy image
    noise_level_model = noise_level_img  # set noise level for model
    model_name = 'drunet_gray'           # set denoiser model, 'drunet_gray' | 'drunet_color'
    testset_name = 'eric'               # set test set,  'bsd68' | 'cbsd68' | 'set12'
    x8 = False                           # default: False, x8 to boost performance
    show_img = False                     # default: False
    border = 0                           # shave boader to calculate PSNR and SSIM

    if 'color' in model_name:
        n_channels = 3                   # 3 for color image
    else:
        n_channels = 1                   # 1 for grayscale image

    model_pool = 'model_zoo'             # fixed
    testsets = 'testsets'                # fixed
    results = 'results'                  # fixed
    task_current = 'dn'                  # 'dn' for denoising
    result_name = testset_name + '_' + task_current + '_' + model_name

    model_path = os.path.join('../denoisers/DPIR/',model_pool, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from denoisers.DPIR.models.network_unet import UNetRes as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    

     # ------------------------------------
     # (1) img_L: low image 
     # ------------------------------------
    img = np.expand_dims(image_in, axis=2)  # HxWx1
    img_L = util.single2tensor4(img)

    print(img_L.dim())  # Output: 2

    img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
    img_L = img_L.to(device)

    # ------------------------------------
    # (2) img_E
    # ------------------------------------

    if not x8 and img_L.size(2)//8==0 and img_L.size(3)//8==0:
        img_E = model(img_L)
    elif not x8 and (img_L.size(2)//8!=0 or img_L.size(3)//8!=0):
        img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
    elif x8:
        img_E = utils_model.test_mode(model, img_L, mode=3)

    img_E = util.tensor2float(img_E)
    util.imsave(img_E, 'denoised.png')

    return img_E


