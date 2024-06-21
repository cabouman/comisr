import numpy as np

def circular_padding(image, padding):

    height, width = image.shape

    # Create an output image with extra padding
    new_height, new_width = height + 2 * padding, width + 2 * padding
    extended_image = np.zeros((new_height, new_width), dtype=image.dtype)
    center_y, center_x = new_height // 2, new_width // 2

    # Place the original image in the center
    start_y, start_x = center_y - height // 2, center_x - width // 2
    extended_image[start_y:start_y + height, start_x:start_x + width] = image

    # Get the border pixels (1 pixel wide border from the original image)
    top_edge = image[0, :]
    bottom_edge = image[-1, :]
    left_edge = image[:, 0]
    right_edge = image[:, -1]

    # Extend the edges radially
    # Top and bottom
    extended_image[start_y - padding:start_y, start_x:start_x + width] = top_edge
    extended_image[start_y + height:start_y + height + padding, start_x:start_x + width] = bottom_edge
    # Left and right
    for i in range(padding):
        extended_image[start_y:start_y + height, start_x - i - 1] = left_edge
        extended_image[start_y:start_y + height, start_x + width + i] = right_edge

    # Corners
    extended_image[start_y - padding:start_y, start_x - padding:start_x] = top_edge[0]
    extended_image[start_y - padding:start_y, start_x + width:start_x + width + padding] = top_edge[-1]
    extended_image[start_y + height:start_y + height + padding, start_x - padding:start_x] = bottom_edge[0]
    extended_image[start_y + height:start_y + height + padding, start_x + width:start_x + width + padding] = bottom_edge[-1]

    return extended_image


def circular_crop(image, cropping=50):
    height, width = image.shape

    # Calculate the new crop rectangle
    left = cropping
    upper = cropping
    right = width - cropping
    lower = height - cropping

    # Crop the image
    cropped_image = image[upper:lower, left:right] 

    return cropped_image

def fft_shift_kernel(kernel, shift_x, shift_y):
    # Step 1: Compute the FFT of the kernel
    kernel_fft = np.fft.fft2(kernel)

    # Step 2: Shift the FFT
    # np.fft.fftshift is used to center the zero frequencies
    # This makes the shift more intuitive and visually comprehensible
    shifted_kernel_fft = np.fft.fftshift(kernel_fft)
    shifted_kernel_fft = np.roll(shifted_kernel_fft, shift_y, axis=0)
    shifted_kernel_fft = np.roll(shifted_kernel_fft, shift_x, axis=1)
    shifted_kernel_fft = np.fft.ifftshift(shifted_kernel_fft)

    # Step 3: Compute the inverse FFT
    shifted_kernel = np.fft.ifft2(shifted_kernel_fft)
    return np.real(shifted_kernel)  # Convert from complex to real
