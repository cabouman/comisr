from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def display_image(image, title='2D Image', cmap='gray', colorbar=True):
    """
    Displays a 2D array as an image using matplotlib.

    Args:
        image (numpy array): The 2D array to be displayed as an image.
        title (str, optional): Title of the image. Defaults to '2D Image'.
        cmap (str, optional): Colormap used to display the image. Defaults to 'gray'.
        colorbar (bool, optional): Flag to indicate whether to display a colorbar. Defaults to True.

    Returns:
        None
    """
    # Determine the global min and max values for consistent scaling across both images
    vmin = image.min()
    vmax = image.max()

    fig, ax = plt.subplots()
    img = ax.imshow(image, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')  # Turn off axis numbering and ticks

    if colorbar:
        plt.colorbar(img, ax=ax)

    plt.show()


def display_images(image1, image2, title1='Image 1', title2='Image 2', colorbar=True):
    """
    Display two grayscale images side by side with optional labels, consistent scaling, and optional colorbar.

    Args:
    image1 (np.ndarray): The first image represented as a NumPy array.
    image2 (np.ndarray): The second image represented as a NumPy array.
    title1 (str): Label for the first image.
    title2 (str): Label for the second image.
    colorbar (bool): If True, display colorbar next to each image.
    """
    # Determine the global min and max values for consistent scaling across both images
    vmin = min(image1.min(), image2.min())
    vmax = max(image1.max(), image2.max())

    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Display first image
    im1 = axes[0].imshow(image1, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[0].axis('off')  # Turn off axis numbers and ticks
    axes[0].set_title(title1)

    # Optionally add a colorbar
    if colorbar:
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Display second image
    im2 = axes[1].imshow(image2, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[1].axis('off')
    axes[1].set_title(title2)

    # Optionally add a colorbar
    if colorbar:
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def display_3images(image1, image2, image3, title1='Image 1', title2='Image 2', title3='Image 3', colorbar=True):
    """
    Display three grayscale images side by side with optional labels, consistent scaling, and optional colorbar.

    Args:
    image1 (np.ndarray): The first image represented as a NumPy array.
    image2 (np.ndarray): The second image represented as a NumPy array.
    imagee (np.ndarray): The second image represented as a NumPy array.
    title1 (str): Label for the first image.
    title2 (str): Label for the second image.
    title3 (str): Label for the second image.
    colorbar (bool): If True, display colorbar next to each image.
    """
    # Determine the global min and max values for consistent scaling across both images
    vmin = min(image1.min(), image2.min(), image3.min())
    vmax = max(image1.max(), image2.max(), image3.max())

    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # Display first image
    im1 = axes[0].imshow(image1, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[0].axis('off')  # Turn off axis numbers and ticks
    axes[0].set_title(title1)

    # Optionally add a colorbar
    if colorbar:
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Display second image
    im2 = axes[1].imshow(image2, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[1].axis('off')
    axes[1].set_title(title2)

    # Optionally add a colorbar
    if colorbar:
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Display third image
    im3 = axes[2].imshow(image3, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[2].axis('off')
    axes[2].set_title(title3)

    # Optionally add a colorbar
    if colorbar:
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def display_3images_pixelval(image1, image2, image3, title1='Image 1', title2='Image 2', title3='Image 3', colorbar=True):
    """
    Display three grayscale images side by side with optional labels, consistent scaling, and optional colorbar.
    The third image is the difference image of image1 and image2, also display third image's pixel value

    Args:
    image1 (np.ndarray): The first image represented as a NumPy array.
    image2 (np.ndarray): The second image represented as a NumPy array.
    imagee (np.ndarray): The second image represented as a NumPy array.
    title1 (str): Label for the first image.
    title2 (str): Label for the second image.
    title3 (str): Label for the second image.
    colorbar (bool): If True, display colorbar next to each image.
    """
    # Determine the global min and max values for consistent scaling across both images
    vmin = min(image1.min(), image2.min(), image3.min())
    vmax = max(image1.max(), image2.max(), image3.max())

    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # Display first image
    im1 = axes[0].imshow(image1, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[0].axis('off')  # Turn off axis numbers and ticks
    axes[0].set_title(title1)

    # Optionally add a colorbar
    if colorbar:
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Display second image
    im2 = axes[1].imshow(image2, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[1].axis('off')
    axes[1].set_title(title2)

    # Optionally add a colorbar
    if colorbar:
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Display third image
    im3 = axes[2].imshow(image3, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[2].axis('off')
    axes[2].set_title(title3)

    # Add pixel values as text annotations
    for i in range(image3.shape[0]):
        for j in range(image3.shape[1]):
            if np.abs(image3[i, j]) > 0.00001:
                plt.text(j, i, f'{image3[i, j]:.3f}', ha='center', va='center', color='red')

    # Optionally add a colorbar
    if colorbar:
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()



def read_png(file_path):
    """
    Reads a PNG image from a specified file path, converts it to a normalized NumPy array.

    Args:
    file_path (str): The path to the PNG file.

    Returns:
    np.ndarray: A NumPy array of the image with pixel values normalized to [0, 1].
                Returns None if an error occurs.
    """
    try:
        # Open an image file
        img = Image.open(file_path)
        # Convert image to a NumPy array
        img_array = np.array(img)
        # Normalize the array to be in the range [0, 1]
        normalized_img_array = img_array.astype(np.float32) / 255.0
        return normalized_img_array
    except IOError:
        print("Error: File does not exist or is not an image.")
        return None


def resize_image(arr, new_shape):
    """
    Resize a 2D floating-point array (interpreted as an image) to new dimensions (N, M) using Pillow.
    The array is expected to be normalized [0, 1] for floating-point inputs.

    Args:
    image (np.ndarray): Input 2D floating-point array (image image).
    high_res_shape (tuple): New width and height as a tuple (M, N).

    Returns:
    np.ndarray: Resized 2D floating-point array.
    """
    # Scale the float array to 0-255 and convert to uint8
    arr_scaled = (arr * 255).astype(np.uint8)
    image = Image.fromarray(arr_scaled)
    # Use LANCZOS for high-quality resampling
    resized_image = image.resize(new_shape, Image.Resampling.LANCZOS)

    # Convert back to float
    resized_array = np.array(resized_image, dtype=np.float32) / 255.0
    return resized_array

