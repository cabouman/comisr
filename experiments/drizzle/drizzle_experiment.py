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
Fraction = 2

frames_number = np.square(Fraction)


# 1. Read in a high resolution image
img = cv2.imread('./input/my_video_frame30.png', 0) 

# Obtain the size of the original image 
[m, n] = img.shape
print('Image Shape:', m, n) 

# Crop image with different offset
img1 = np.zeros(shape=(frames_number,600,1080), dtype=img.dtype)
img_drizzed = np.zeros(shape=(frames_number,600,1080), dtype=img.dtype)

#image1 = np.zeros(shape=(frames_number,180,300), dtype=img.dtype)
#img_drizzed = np.zeros(shape=(frames_number,180,300), dtype=img.dtype)


# Crop images with random shifts
for i in range(frames_number):
    img1[i] = image_processing.crop_image(img, offset_max = Fraction)
    cv2.imwrite('./output/img1_cropped_' + str(i) + '_.png', img1[i]) 



""" # Crop images with fixed shifts
if Fraction == 3:
    shift_list = [-1, 0, 1]
if Fraction == 2:
    shift_list = [-1, 1]

frame_count = 0
for x_shift in shift_list:
    for y_shift in shift_list:
            # print("Frame, Fixed x_shift and y_shift: ", frame_count, x_shift, y_shift)
            #image1[frame_count] = image_processing.crop_image_fixed(img, offset_max = Fraction, offset_x = x_shift*Fraction, offset_y = y_shift*Fraction)
            image1[frame_count] = image_processing.crop_image_fixed(img, offset_max = Fraction, offset_x = x_shift*1, offset_y = y_shift*1)
            cv2.imwrite('./output/img1_cropped_fix_' + str(frame_count) + '_.png', image1[frame_count])
            frame_count = frame_count + 1  """


[width, height] = img1[0].shape
print('cropped image size:', width,height)

#cv2.imshow("cropped", image1)
#cv2.waitKey(0)

""" # Down sample input image - no longer used. Use opencv libraries
# pix_img1 = image1.load()
#img1_down = image_processing.down_sample(image1, f=2)
#img2_down = image_processing.down_sample(image2, f=2)

#cv2.imshow("down sampled image 1", img1_down)
#cv2.imshow("down sampled image 2", img2_down)

#cv2.waitKey(0)
#cv2.destroyAllWindows() """

# resize image
scale_percent = 1/Fraction # percent of original size
width = int(img1[0].shape[1] * scale_percent)
height = int(img1[0].shape[0] * scale_percent)
dim = (width, height)

img_down = np.zeros(shape=(frames_number, height,width), dtype=img.dtype)
img_down_align = np.zeros(shape=(frames_number, height,width), dtype=img.dtype)

for i in range(frames_number):
    img_down[i] = cv2.resize(img1[i], dim, interpolation = cv2.INTER_AREA)
    #img_down_align[i] = image_processing.image_align(img_down[i], img_down[0])
    img_down_align[i] = img_down[i]

cv2.imwrite('img_down_0.png', img_down[0]) 

# Obtain the size of the original image 
[m, n] = img_down[0].shape
print('Downsampled Image Shape:', m, n) 

# drizzle image
start_time = time.time()
[img_drizzed[0], areamap] = image_processing.drizzle_trio(img_down[0], f=Fraction, a = 0.5, weight = 2)
print("---The running time for up sampling is %s seconds ---" % (time.time() - start_time))
cv2.imwrite('img_drizzle0.png', img_drizzed[0]) 

for i in range(1,frames_number):
    start_time = time.time()
    [img_drizzed[i], areamap, weightmap] = image_processing.drizzle(img_down[0], img_down[i], f=Fraction, a = 0.5, weight = 1, I = img_drizzed[i-1])
    print("---The running time for one drizzle is %s seconds ---" % (time.time() - start_time))
    cv2.imwrite('img_drizzle_' + str(i) + '.png', img_drizzed[i])

#[img_drizzle3, areamap, weightmap] = image_processing.drizzle(img_down[0], img_down_align[2], f=Fraction, a = 0.5, weight = 1, I = img_drizzle2)
#[img_drizzle4, areamap, weightmap] = image_processing.drizzle(img_down[0], img_down_align[3], f=Fraction, a = 0.5, weight = 1, I = img_drizzle3)
#[img_drizzle5, areamap, weightmap] = image_processing.drizzle(img_down[0], img_down_align[4], f=Fraction, a = 0.5, weight = 1, I = img_drizzle3)


#cv2.imwrite('img_drizzle1.png', img_drizzed[1]) 
#cv2.imwrite('img_drizzle2.png', img_drizzed[2]) 
#cv2.imwrite('img_drizzle3.png', img_drizzed[3])
cv2.imwrite('img_drizzle_final.png', img_drizzed[frames_number-1]) 

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





