from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def display_image(data, title='2D Image', cmap='gray', colorbar=True):
    """
    Displays a 2D array as an image using matplotlib.

    Args:
        data (numpy array): The 2D array to be displayed as an image.
        title (str, optional): Title of the image. Defaults to '2D Image'.
        cmap (str, optional): Colormap used to display the image. Defaults to 'gray'.
        colorbar (bool, optional): Flag to indicate whether to display a colorbar. Defaults to True.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    img = ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')  # Turn off axis numbering and ticks

    if colorbar:
        plt.colorbar(img, ax=ax)

    plt.show()


def display_images(img1, img2, label1='Image 1', label2='Image 2'):
    """
    Display two grayscale images side by side with optional labels and consistent scaling.

    Args:
    img1 (jnp.ndarray): The first image represented as a JAX array.
    img2 (jnp.ndarray): The second image represented as a JAX array.
    label1 (str): Label for the first image.
    label2 (str): Label for the second image.
    """
    # Convert JAX arrays to NumPy arrays for display
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    # Determine the global min and max values for consistent scaling across both images
    vmin = min(img1_np.min(), img2_np.min())
    vmax = max(img1_np.max(), img2_np.max())

    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display first image
    axes[0].imshow(img1_np, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[0].axis('off')  # Turn off axis numbers and ticks
    axes[0].set_title(label1)

    # Display second image
    axes[1].imshow(img2_np, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[1].axis('off')
    axes[1].set_title(label2)

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
    image (np.ndarray): Input 2D floating-point array (image data).
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
