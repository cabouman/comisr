# Implement equations (2)-(5) using known translations from simulated image stack – 
# take a single image and translate it by known amounts to generate a stack and 
# then recover the original.  
# Coarse grid and fine grid are the same for step 1.

import numpy as np
import cv2


# 1. Read in a high resolution image
img = cv2.imread('./input/my_video_frame30.png', 0) 

# Obtain the size of the original image 
[m, n] = img.shape
print('Image Shape:', m, n) 

# Crop the image from original input image from random offset
def crop_image(img, offset_max):
    x_shift = np.random.randint(0,offset_max)
    y_shift = np.random.randint(0,offset_max)
    print("x_shift and y_shift: ", x_shift, y_shift)

    # Test the code
    DEBUG = 0

    if DEBUG == 1: 
        left = 580 - offset_max + 4
        top = 280 + 1
        right = left + 8
        bottom = top + 8
    else: 
        left = 200 - offset_max + x_shift
        top = 0 + y_shift
        right = left + 1080
        bottom = top + 600

    # Cropped image of above dimension
    #img_croped = img.crop((left, top, right, bottom))
    img_croped = img[top:bottom, left:right]
    return img_croped 

# Down sample the original image into with a fraction of f 
# This function is not used. 
def down_sample(im1, f):
    [m, n] = im1.shape
    # print(m,n)
    # Create an image with zero values
    im2 = np.zeros((m//f, n//f), im1.dtype) 
    [width, height] = im2.shape
    print("im2 size:", width, height)

    # Assign the down sampled values from the original 
    # image according to the down sampling frequency. 
    # For example, if the down sampling rate f=2, take 
    # pixel values from alternate rows and columns 
    # and assign them in the matrix created above 
    for i in range(0, m, f): 
        for j in range(0, n, f): 
            try:
                im2[i//f][j//f] = im1[i][j] 
            except IndexError: 
                pass
    return im2


# Up sampling 
def drizzle(im2, f, a, weight):
    [m, n] = im2.shape
    # Create matrix of zeros to store the upsampled image 
    im3 = np.zeros((m*f, n*f), dtype=im2.dtype)
    areaMap = np.zeros((m*f, n*f), dtype=im2.dtype)
    weightMap = np.ones((m*f, n*f), dtype=np.double)


    for i in range(0,m):
        for j in range(0,n):
            Bx1 = i*f 
            By1 = j*f
            Bx2 = (i+1)*f
            By2 = (j+1)*f
            
            offset = f*(1-np.sqrt(a))/2
            bx1 = Bx1 + offset
            by1 = By1 + offset
            bx2 = Bx2 - offset
            by2 = By2 - offset

            for k1 in range (0,f):
                for k2 in range(0,f):
                    # check the overlap area
                    dx = np.minimum(Bx1+k1+1, bx2) - np.maximum(Bx1+k1, bx1)
                    dy = np.minimum(By1+k2+1, by2) - np.maximum(By1+k2, by1)
                    area = np.maximum(0, dx) * np.maximum(0, dy)
                    areaMap[i*f+k1, j*f+k2] = round(area*100)
                    weightMap[i*f+k1, j*f+k2] = areaMap[i*f+k1, j*f+k2]*weight + 1
                    im3[i*f+k1, j*f+k2] = round(im2[i,j]*area*weight)

    return im3, areaMap

# Crop image with different offset
img1 = crop_image(img, offset_max = 50)
img2 = crop_image(img, offset_max = 50)
[width, height] = img1.shape
print('cropped image size:', width,height)

cv2.imwrite('img1_cropped.png', img1) 
#cv2.imshow("cropped", img1)
#cv2.waitKey(0)

# Down sample input image 
# pix_img1 = img1.load()
#img1_down = down_sample(img1, f=2)
#img2_down = down_sample(img2, f=2)

#cv2.imshow("down sampled image 1", img1_down)
#cv2.imshow("down sampled image 2", img2_down)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

# resize image
f = 4
scale_percent = 1/f # percent of original size
width = int(img1.shape[1] * scale_percent)
height = int(img1.shape[0] * scale_percent)
dim = (width, height)
img1_down = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

# drizzle image
[img_drizzed, areamap] = drizzle(img1_down, f, a = 0.5, weight = 1)
cv2.imwrite('img1_down.png', img1_down) 
cv2.imwrite('drizzedimage4.png', img_drizzed) 
cv2.imwrite('areaMap4.png', areamap)

[img_drizzed, areamap] = drizzle(img1_down, f=3, a = 0.5, weight = 1)
cv2.imwrite('drizzedimage3.png', img_drizzed) 
cv2.imwrite('areaMap3.png', areamap)

cv2.imshow("drizzedimage 2", img_drizzed)









