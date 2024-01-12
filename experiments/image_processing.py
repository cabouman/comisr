# Image aligment 

import cv2 
import numpy as np 
from numba import njit


def image_align(img1, img2):

    ## Open the image files. 
    #img1_color = cv2.imread("align.jpg") # Image to be aligned. 
    #img2_color = cv2.imread("ref.jpg") # Reference image. 

    # Convert to grayscale. 
    #img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
    #img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) 
    height, width = img2.shape 

    # Create ORB detector with 5000 features. 
    orb_detector = cv2.ORB_create(5000) 

    # Find keypoints and descriptors. 
    # The first arg is the image, second arg is the mask 
    # (which is not required in this case). 
    kp1, d1 = orb_detector.detectAndCompute(img1, None) 
    kp2, d2 = orb_detector.detectAndCompute(img2, None) 

    # Match features between the two images. 
    # We create a Brute Force matcher with 
    # Hamming distance as measurement mode. 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 

    # Match the two sets of descriptors. 
    matches = matcher.match(d1, d2) 

    # Sort matches on the basis of their Hamming distance. 
    # matches.sort(key = lambda x: x.distance)   # Yufang: this line does not work. 
    matches = sorted(matches, key=lambda x: x.distance)

    # Take the top 90 % matches forward. 
    matches = matches[:int(len(matches)*0.9)] 
    no_of_matches = len(matches) 

    # Define empty matrices of shape no_of_matches * 2. 
    p1 = np.zeros((no_of_matches, 2)) 
    p2 = np.zeros((no_of_matches, 2)) 

    for i in range(len(matches)): 
        p1[i, :] = kp1[matches[i].queryIdx].pt 
        p2[i, :] = kp2[matches[i].trainIdx].pt 

    # Find the homography matrix. 
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 

    # Use this matrix to transform the 
    # colored image wrt the reference image. 
    transformed_img = cv2.warpPerspective(img1, 
					homography, (width, height)) 

    # Save the output. 
    #cv2.imwrite('output.jpg', transformed_img) 
    return transformed_img

# Crop the image from original input image from random offset
@njit
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
@njit
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
@njit
def drizzle_trio(im2, f, a, weight):
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
            
            # offset = f*(1-np.sqrt(a))/2 
            offset = f*(1-a)/2
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

            #areablock = sum(sum(areaMap[i*f:(i+1)*f-1, j*f:(j+1)*f-1]))

            '''
            for k1 in range (0,f):
                for k2 in range(0,f):
                    im3[i*f+k1, j*f+k2] = round(im2[i,j]*areaMap[i*f+k1, j*f+k2]/areablock*weight)
'''

    return im3, areaMap

def drizzle_addto(imd, im0, f, a, weightd, weight0, I):
    # imd: reference image; im0: to be drizzled to
    [m, n] = imd.shape
    [m0, n0] = im0.shape
    # Create matrix of zeros to store the upsampled image 
    im3 = np.zeros((m*f, n*f), dtype=imd.dtype)
    areaMap = np.zeros((m*f, n*f), dtype=imd.dtype)
    WeightMap = np.ones((m*f, n*f), dtype=np.double)

    for i in range(0,m):
        for j in range(0,n):
            Bx1 = i*f 
            By1 = j*f
            Bx2 = (i+1)*f
            By2 = (j+1)*f
            
            # offset = f*(1-np.sqrt(a))/2 
            offset = f*(1-a)/2
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
                    WeightMap[i*f+k1, j*f+k2] = areaMap[i*f+k1, j*f+k2]*weightd
                    #im3[i*f+k1, j*f+k2] = round(im0[i,j]*area*weightd[i*f+k1, j*f+k2])
                    #im3[i*f+k1, j*f+k2] = im3[i*f+k1, j*f+k2] + imd[i,j]
                    im3[i*f+k1, j*f+k2] = (imd[i, j]*area*weightd + im0[i, j]*weight0)/WeightMap[i*f+k1, j*f+k2]
    # update I
    #I = (I + im3)/2

    return im3, areaMap, WeightMap

@njit
def drizzle(imd, im0, f, a, weight, I):
    # imd: reference image; im0: to be drizzled to
    [m, n] = im0.shape
    # Create matrix of zeros to store the upsampled image 
    im3 = np.zeros((m*f, n*f), dtype=np.float64)
    areaMap = np.zeros((m*f, n*f), dtype=im0.dtype)
    WeightMap = np.ones((m*f, n*f), dtype=np.double)

    for i in range(0,m):
        for j in range(0,n):
            Bx1 = i*f 
            By1 = j*f
            Bx2 = (i+1)*f
            By2 = (j+1)*f
            
            # offset = f*(1-np.sqrt(a))/2 
            offset = f*(1-a)/2
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
                    WeightMap[i*f+k1, j*f+k2] = area*weight + 1
                    im3[i*f+k1, j*f+k2] = round(im0[i,j]*area*weight)
                    #im3[i*f+k1, j*f+k2] = im3[i*f+k1, j*f+k2] + imd[i,j]
                    im3[i*f+k1, j*f+k2] = (im3[i*f+k1, j*f+k2] + imd[i,j])/WeightMap[i*f+k1, j*f+k2] 

    # update I
    #I = (I + im3)/2

    return im3, areaMap, WeightMap
