# Implement equations (2)-(5) using known translations from simulated image stack – 
# take a single image and translate it by known amounts to generate a stack and 
# then recover the original.  
# Coarse grid and fine grid are the same for step 1.

import numpy as np
import cv2
import time

import image_processing

# add liabriries that do de-converlution
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration

# define number of frames
frames_number = 4

# 1. Read in a high resolution image
img = cv2.imread('./input/my_video_frame30.png', 0) 

# Obtain the size of the original image 
[m, n] = img.shape
print('Image Shape:', m, n) 

# Crop image with different offset
img1 = np.zeros(shape=(frames_number,600,1080), dtype=img.dtype)

for i in range(frames_number):
    img1[i] = image_processing.crop_image(img, offset_max = 8)
    #img1 = image_processing.crop_image(img, offset_max = 8)
    #img2 = image_processing.crop_image(img, offset_max = 8)
    #img3 = image_processing.crop_image(img, offset_max = 8)
    #img4 = image_processing.crop_image(img, offset_max = 8)

[width, height] = img1[0].shape
print('cropped image size:', width,height)

cv2.imwrite('img1_cropped.png', img1[0]) 
#cv2.imshow("cropped", img1)
#cv2.waitKey(0)

""" # Down sample input image 
# pix_img1 = img1.load()
#img1_down = image_processing.down_sample(img1, f=2)
#img2_down = image_processing.down_sample(img2, f=2)

#cv2.imshow("down sampled image 1", img1_down)
#cv2.imshow("down sampled image 2", img2_down)

#cv2.waitKey(0)
#cv2.destroyAllWindows() """

# resize image
f = 2
scale_percent = 1/f # percent of original size
width = int(img1[0].shape[1] * scale_percent)
height = int(img1[0].shape[0] * scale_percent)
dim = (width, height)

img_down = np.zeros(shape=(frames_number, height,width), dtype=img.dtype)
img_down_align = np.zeros(shape=(frames_number, height,width), dtype=img.dtype)

for i in range(frames_number):
    img_down[i] = cv2.resize(img1[i], dim, interpolation = cv2.INTER_AREA)
    img_down_align[i] = image_processing.image_align(img_down[i], img_down[0])


# drizzle image
start_time = time.time()
[img_up1, areamap] = image_processing.drizzle_trio(img_down[0], f=2, a = 0.5, weight = 2)
print("---The running time for up sampling is %s seconds ---" % (time.time() - start_time))

[img_up2, areamap] = image_processing.drizzle_trio(img_down[1], f=2, a = 0.5, weight = 2)
[img_up3, areamap] = image_processing.drizzle_trio(img_down[2], f=2, a = 0.5, weight = 3)
[img_up4, areamap] = image_processing.drizzle_trio(img_down[3], f=2, a = 0.5, weight = 4)

img_drizzle = (img_up1 + img_up2 + img_up3 + img_up4)/4
cv2.imwrite('img_drizzle.png', img_drizzle) 

start_time = time.time()
[img_drizzle2, areamap, weightmap] = image_processing.drizzle(img_down[0], img_down_align[1], f=2, a = 0.5, weight = 1, I = img_up1)
print("---The running time for each drizzle is %s seconds ---" % (time.time() - start_time))

[img_drizzle3, areamap, weightmap] = image_processing.drizzle(img_down[0], img_down_align[2], f=2, a = 0.5, weight = 1, I = img_drizzle2)
[img_drizzle4, areamap, weightmap] = image_processing.drizzle(img_down[0], img_down_align[3], f=2, a = 0.5, weight = 1, I = img_drizzle3)

cv2.imwrite('img_drizzle1.png', img_up1) 
cv2.imwrite('img_drizzle2.png', img_drizzle2) 
cv2.imwrite('img_drizzle3.png', img_drizzle3) 
cv2.imwrite('img_drizzle4.png', img_drizzle4) 
cv2.imwrite('areaMap4.png', areamap)

#[img_drizzed, areamap] = drizzle(img1_down, f=3, a = 0.5, weight = 1)
#cv2.imwrite('drizzedimage3.png', img_drizzed) 
#cv2.imwrite('areaMap3.png', areamap)
#cv2.imshow("drizzedimage 2", img_drizzed)

# deconvelution 
psf = np.ones((5, 5)) / 25
astro_img = conv2(img, psf, 'same')

# Restore Image using Richardson-Lucy algorithm
deconvolved_img = restoration.richardson_lucy(astro_img, psf, num_iter=30)
cv2.imwrite('deconvolved_img.png', deconvolved_img) 





