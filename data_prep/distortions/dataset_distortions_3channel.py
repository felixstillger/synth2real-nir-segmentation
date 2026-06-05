#!/usr/bin/env python

from scipy.ndimage import gaussian_filter
from scipy import fftpack as fp
from skimage.io import imread
from PIL import Image
import numpy as np
import os
import shutil
import yaml
import argparse
import cv2

###########################################################
#   IMAGE IO
###########################################################

def imload_rgb(path):
    """Load and return an RGB image in the range [0, 1]."""
    return imread(path).astype(np.float64) / 255.0


def imload_mask(path):
    """Load and return a segmentation mask as uint8 grayscale."""
    img = Image.open(path)
    mask = np.array(img)
    # Ensure it's 2D (grayscale/indexed)
    if mask.ndim == 3:
        mask = mask[:, :, 0]  # Take first channel if RGB
    return mask.astype(np.uint8)


def save_img(image, imgname, use_JPEG=False):
    """Save image as either .jpeg or .png"""
    format = "JPEG" if use_JPEG else 'PNG'

    image = np.clip(image, 0.0, 1.0)  # Ensure values are within [0,1]
    image = (image * 255.0).astype("uint8")
    Image.fromarray(image).save(imgname, format)


def save_mask(mask, maskname):
    """Save segmentation mask as uint8 grayscale PNG."""
    mask = mask.astype(np.uint8)
    Image.fromarray(mask, mode='L').save(maskname, 'PNG')


###########################################################
#   IMAGE MANIPULATION
#
# In general, images are handled as follows:
# - datatype: numpy.ndarray
# - numpy dtype: float64
# - channels: RGB (3 channels)
# - range: [0, 1]
#
###########################################################


def adjust_contrast(image, contrast_level):
    """Return the image scaled to a certain contrast level in [0, 1].

    parameters:
    - image: a numpy.ndarray with shape (H, W, 3)
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    """

    assert 0.0 <= contrast_level <= 1.0, "contrast_level must be within [0, 1]."

    # Apply contrast adjustment per channel
    return (1 - contrast_level) / 2.0 + image * contrast_level


def uniform_noise(image, width, contrast_level, rng):
    """Apply uniform noise to an RGB image after adjusting contrast.

    parameters:
    - image: a numpy.ndarray with shape (H, W, 3)
    - width: a scalar indicating width of additive uniform noise
             -> noise will be in range [-width, width]
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    # Adjust contrast per channel
    image = adjust_contrast(image, contrast_level)

    return apply_uniform_noise_rgb(image, -width, width, rng)


def salt_and_pepper_noise(image, p, contrast_level, rng):
    """Apply salt and pepper noise to an RGB image after adjusting contrast.

    parameters:
    - image: a numpy.ndarray with shape (H, W, 3)
    - p: a scalar indicating probability of white and black pixels, in [0, 1]
    - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    assert 0 <= p <= 1, "Probability p must be within [0, 1]."

    # Adjust contrast per channel
    image = adjust_contrast(image, contrast_level)
    assert image.ndim == 3 and image.shape[2] == 3, "Image must have 3 channels."

    # Generate noise for each channel
    salt = rng.uniform(size=image[:,:,0].shape) >= 1 - p / 2
    pepper = rng.uniform(size=image[:,:,0].shape) < p / 2

    # Apply salt and pepper
    for c in range(image.shape[2]):
        image[:,:,c][salt] = 1.0
        image[:,:,c][pepper] = 0.0

    # Ensure values are within [0, 1]
    image = np.clip(image, 0, 1)

    assert is_in_bounds(image, 0, 1), "Values <0 or >1 occurred"

    return image


def high_pass_filter(image, std, mean_color):
    """Apply a Gaussian high pass filter to an RGB image by calculating Highpass(image) = image - Lowpass(image).

    parameters:
    - image: a numpy.ndarray with shape (H, W, 3)
    - std: a scalar providing the Gaussian low-pass filter's standard deviation
    - mean_color: list of mean colors [mean_r, mean_g, mean_b]
    """

    # Convert to RGB channels and apply high-pass filter per channel
    new_image = np.zeros_like(image)
    for c in range(3):
        gauss = gaussian_filter(image[:, :, c], std, mode='constant', cval=mean_color[c])
        high_pass = image[:, :, c] - gauss
        # Add mean difference to retain image statistics
        mean_diff = mean_color[c] - np.mean(high_pass)
        high_pass += mean_diff
        # Clip values to [0,1]
        high_pass = np.clip(high_pass, 0, 1)
        new_image[:, :, c] = high_pass

    return new_image


def low_pass_filter(image, std, mean_color):
    """Apply a Gaussian low-pass filter to an RGB image.

    parameters:
    - image: a numpy.ndarray with shape (H, W, 3)
    - std: a scalar providing the Gaussian low-pass filter's standard deviation
    - mean_color: list of mean colors [mean_r, mean_g, mean_b]
    """

    # Apply low-pass filter per channel
    new_image = np.zeros_like(image)
    for c in range(3):
        low_pass = gaussian_filter(image[:, :, c], std, mode='constant', cval=mean_color[c])
        low_pass = np.clip(low_pass, 0, 1)
        new_image[:, :, c] = low_pass

    return new_image


def phase_scrambling(image, width):
    """Apply random shifts to an image's frequencies' phases in the Fourier domain per channel.

    parameter:
    - image: a numpy.ndarray with shape (H, W, 3)
    - width: maximal width of the random phase shifts
    """

    scrambled_image = np.zeros_like(image)
    for c in range(3):
        scrambled_image[:, :, c] = scramble_phases(image[:, :, c], width)
    return scrambled_image


def elastic_transform(image, alpha, sigma, random_state=None):
    """Apply elastic deformation to an image.
    
    parameters:
    - image: a numpy.ndarray (can be RGB or grayscale)
    - alpha: scaling factor for deformation
    - sigma: standard deviation for Gaussian filter
    - random_state: np.random.RandomState for reproducibility
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    interp = cv2.INTER_NEAREST
    remapped = cv2.remap(image, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REFLECT)
    return remapped


def swirl_transform(image, strength, radius=None):
    """Apply swirl deformation to an image.
    
    parameters:
    - image: a numpy.ndarray (can be RGB or grayscale)
    - strength: strength of the swirl effect
    - radius: radius of effect (default: min(h,w)/2)
    """
    h, w = image.shape[:2]
    if radius is None:
        radius = min(h, w) / 2.0
    cy, cx = h / 2.0, w / 2.0
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dy = y - cy
    dx = x - cx
    r = np.sqrt(dx*dx + dy*dy)
    theta = np.arctan2(dy, dx)
    amount = strength * np.exp(-(r**2) / (2.0 * (radius**2)))
    theta_t = theta + amount
    x_t = cx + r * np.cos(theta_t)
    y_t = cy + r * np.sin(theta_t)
    map_x = x_t.astype(np.float32)
    map_y = y_t.astype(np.float32)
    interp = cv2.INTER_NEAREST
    return cv2.remap(image, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REFLECT)


###########################################################
#   HELPER FUNCTIONS
###########################################################

def apply_uniform_noise_rgb(image, low, high, rng=None):
    """Apply uniform noise to an RGB image, clip outside values to 0 and 1.

    parameters:
    - image: a numpy.ndarray with shape (H, W, 3)
    - low: lower bound of noise within [low, high)
    - high: upper bound of noise within [low, high)
    - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
    """

    if rng is None:
        noise = np.random.uniform(low=low, high=high, size=image.shape)
    else:
        noise = rng.uniform(low=low, high=high, size=image.shape)

    image_noisy = image + noise
    image_noisy = np.clip(image_noisy, 0, 1)

    assert is_in_bounds(image_noisy, 0, 1), "Values <0 or >1 occurred"

    return image_noisy


def is_in_bounds(mat, low, high):
    """Return whether all values in 'mat' fall between low and high.

    parameters:
    - mat: a numpy.ndarray 
    - low: lower bound (inclusive)
    - high: upper bound (inclusive)
    """

    return np.all((mat >= low) & (mat <= high))


def scramble_phases(channel, width):
    """Apply random shifts to an image channel's phases in the Fourier domain.

    parameter:
    - channel: a 2D numpy.ndarray
    - width: maximal width of the random phase shifts (in degrees)
    """

    # Convert width from degrees to radians
    width_rad = np.deg2rad(width)

    # Fourier Forward Transform and shift to center
    f = fp.fft2(channel)
    f_shifted = fp.fftshift(f)

    # Get amplitudes and phases
    f_amp = np.abs(f_shifted)
    f_phase = np.angle(f_shifted)

    # Create random phase shifts within [-width_rad, width_rad]
    phase_shifts = np.random.uniform(-width_rad, width_rad, size=f_phase.shape)

    # Apply phase shifts
    f_phase += phase_shifts

    # Reconstruct the Fourier transform with new phases
    f_new = f_amp * np.exp(1j * f_phase)

    # Inverse shift and perform Fourier Backward Transformation
    f_new_shifted = fp.ifftshift(f_new)
    new_channel = fp.ifft2(f_new_shifted).real

    # Clip values to [0,1]
    new_channel = np.clip(new_channel, 0, 1)

    return new_channel


###########################################################
#   MAIN FUNCTION TO CREATE DISTORTED DATASETS
###########################################################

def compute_dataset_mean_color(root_paths, extensions=('.png', '.jpg', '.jpeg')):
    """Compute per-channel mean color across images found under the given root paths."""
    total_sum = np.zeros(3, dtype=np.float64)
    total_count = 0
    exts = tuple(ext.lower() for ext in extensions)
    paths = []
    for root_path in root_paths:
        for root, _, files in os.walk(root_path):
            for f in files:
                if f.lower().endswith(exts):
                    paths.append(os.path.join(root, f))

    if len(paths) == 0:
        raise ValueError("No image files found to compute mean color.")
    for p in paths:
        img = imload_rgb(p)
        total_sum += img.reshape(-1, 3).sum(axis=0)
        total_count += img.shape[0] * img.shape[1]
    mean = (total_sum / total_count).tolist()
    return mean


def create_distorted_cityscapes(csroot, output_root,
                                distortions_config=None,
                                use_JPEG=False,
                                mean_color=None,
                                dataset_name=None,
                                image_subdir=None,
                                extensions=('.png', '.jpg', '.jpeg'),
                                is_mask=False):
    """
    Create distorted variations of a dataset.

    Parameters:
    - csroot: str, path to the dataset root.
    - output_root: str, path where the distorted datasets will be saved.
    - distortions_config: dict, configuration of distortions and their parameter lists.
    - use_JPEG: bool, whether to save images as JPEG (True) or PNG (False).
    - mean_color: list of mean colors [mean_r, mean_g, mean_b] to use for filters.
                  If None, will be computed from images found under csroot (or image_subdir).
    - dataset_name: optional name to prefix output folders. Defaults to the basename of csroot.
    - image_subdir: optional subdirectory (string or list) under csroot where images live (e.g., 'images' or ['val','train']).
    - extensions: tuple of file extensions to consider as images.
    - is_mask: bool, if True treat input as single-channel grayscale masks rather than RGB images.
    """

    if distortions_config is None:
        print("No distortions specified. Exiting.")
        return

    if dataset_name is None:
        dataset_name = os.path.basename(os.path.normpath(csroot)) or 'dataset'

    # Determine image subdirectories (as list)
    if image_subdir:
        if isinstance(image_subdir, (list, tuple)):
            image_subdirs = image_subdir
        else:
            image_subdirs = [image_subdir]
    else:
        image_subdirs = ['.']  # Current directory

    # Build full paths for image subdirs
    image_roots = [os.path.join(csroot, sd) for sd in image_subdirs]

    if mean_color is None and not is_mask:
        # Only compute mean color for RGB images (not needed for masks or geometric distortions)
        mean_color = compute_dataset_mean_color(image_roots, extensions=extensions)
        print(f"Computed mean color for dataset '{dataset_name}': {mean_color}")

    exts = tuple(ext.lower() for ext in extensions)

    # Iterate over each distortion type
    for distortion_name, params_list in distortions_config.items():
        
        for params in params_list:
            # Construct the output directory name based on distortion and parameters
            if distortion_name == 'contrast':
                params_str = f"contrast{params.get('contrast_level')}"
            elif distortion_name == 'uniform_noise':
                params_str = f"contrast{params.get('contrast_level', 0.3)}_uniform_noise{params.get('width')}"
            elif distortion_name in ['low_pass_filter', 'high_pass_filter']:
                std = params.get('std')
                if isinstance(std, float) and np.isinf(std):
                    params_str = "stdinf"
                else:
                    params_str = f"std{std}"
            elif distortion_name == 'phase_scrambling':
                params_str = f"width{int(params.get('width',0))}"
            elif distortion_name == 'elastic_transform':
                params_str = f"alpha{params.get('alpha')}_sigma{params.get('sigma')}"
            elif distortion_name == 'swirl_transform':
                params_str = f"strength{params.get('strength')}_radius{params.get('radius', 'auto')}"
            else:
                # For any other distortion names, construct params_str generically
                params_str = "_".join([f"{key}{value}" for key, value in params.items()])

            distorted_root = os.path.join(output_root, f"{dataset_name}_{distortion_name}_{params_str}")
            print(f"Creating distorted dataset: {distorted_root}")

            # Traverse each image directory
            for img_subdir, image_root in zip(image_subdirs, image_roots):
                for root, dirs, files in os.walk(image_root):
                    # Compute relative path from the image root
                    rel_path = os.path.relpath(root, image_root)

                    # Image output mirrors image tree: <distorted_root>/<img_subdir>/<rel_path>/<file>
                    if rel_path == '.':
                        output_dir = os.path.join(distorted_root, img_subdir)
                        rel_file_path = ''  # For computing mask path
                    else:
                        output_dir = os.path.join(distorted_root, img_subdir, rel_path)
                        rel_file_path = rel_path

                    os.makedirs(output_dir, exist_ok=True)

                    for file in files:
                        if file.lower().endswith(exts):
                            original_file_path = os.path.join(root, file)
                            output_file_path = os.path.join(output_dir, file)

                            try:
                                if is_mask:
                                    image = imload_mask(original_file_path)
                                    # Expand to 3D for consistency (will be squeezed back before saving)
                                    if image.ndim == 2:
                                        image = np.expand_dims(image, axis=2)
                                else:
                                    image = imload_rgb(original_file_path)
                            except Exception as e:
                                print(f"Skipping {original_file_path}: could not load image: {e}")
                                continue

                            # Apply the specified distortion
                            try:
                                if is_mask and distortion_name not in ['elastic_transform', 'swirl_transform']:
                                    # Do not apply appearance distortions to masks
                                    distorted_image = image

                                elif distortion_name == 'contrast':
                                    contrast_level = params.get('contrast_level', 1.0)
                                    # Optionally, convert to grayscale if needed
                                    # Here, we apply contrast adjustment per channel
                                    distorted_image = adjust_contrast(image, contrast_level)

                                elif distortion_name == 'uniform_noise':
                                    width = params.get('width', 0.0)
                                    contrast_level = params.get('contrast_level', 0.3)  # Adjustable contrast level
                                    rng_seed = params.get('rng_seed', 42)
                                    rng = np.random.RandomState(seed=rng_seed)
                                    distorted_image = uniform_noise(image, width, contrast_level, rng)

                                elif distortion_name == 'salt_and_pepper_noise':
                                    p = params.get('p', 0.05)
                                    contrast_level = params.get('contrast_level', 0.3)
                                    rng_seed = params.get('rng_seed', 42)
                                    rng = np.random.RandomState(seed=rng_seed)
                                    distorted_image = salt_and_pepper_noise(image, p, contrast_level, rng)

                                elif distortion_name == 'low_pass_filter':
                                    std = params.get('std', 0.0)
                                    if std == 0.0:
                                        distorted_image = image  # Original image
                                    else:
                                        distorted_image = low_pass_filter(image, std, mean_color)

                                elif distortion_name == 'high_pass_filter':
                                    std = params.get('std', 0.0)
                                    if isinstance(std, float) and np.isinf(std):
                                        distorted_image = image  # Original image
                                    # elif std == 0.0:
                                    #     distorted_image = image  # Original image
                                    else:
                                        distorted_image = high_pass_filter(image, std, mean_color)

                                elif distortion_name == 'phase_scrambling':
                                    width = params.get('width', 0)
                                    distorted_image = phase_scrambling(image, width)

                                elif distortion_name == 'elastic_transform':
                                    alpha = params.get('alpha', 1.0)
                                    sigma = params.get('sigma', 1.0)
                                    rng_seed = params.get('rng_seed', 42)
                                    rng = np.random.RandomState(seed=rng_seed)
                                    if is_mask:
                                        image_uint8 = np.asarray(image, dtype=np.uint8)
                                    else:
                                        image_uint8 = (np.asarray(image) * 255.0).astype(np.uint8)

                                    # Precompute displacement field
                                    shape = image_uint8.shape[:2]
                                    dx = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
                                    dy = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
                                    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                                    map_x = (x + dx).astype(np.float32)
                                    map_y = (y + dy).astype(np.float32)

                                    distorted_uint8 = cv2.remap(image_uint8, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
                                    if is_mask:
                                        distorted_image = distorted_uint8
                                    else:
                                        distorted_image = distorted_uint8.astype(np.float64) / 255.0

                                elif distortion_name == 'swirl_transform':
                                    strength = params.get('strength', 1.0)
                                    radius = params.get('radius')
                                    # Convert image to uint8 for cv2.remap
                                    if is_mask:
                                        image_uint8 = np.asarray(image, dtype=np.uint8)
                                    else:
                                        image_uint8 = (np.asarray(image) * 255.0).astype(np.uint8)
                                    distorted_uint8 = swirl_transform(image_uint8, strength, radius)
                                    if is_mask:
                                        distorted_image = distorted_uint8
                                    else:
                                        distorted_image = distorted_uint8.astype(np.float64) / 255.0

                                else:
                                    print(f"Unknown distortion: {distortion_name}. Skipping.")
                                    continue

                                # Save the distorted image
                                if is_mask:
                                    # For masks, squeeze back to 2D and save as grayscale
                                    if distorted_image.ndim == 3 and distorted_image.shape[2] == 1:
                                        distorted_image = distorted_image[:, :, 0]
                                    save_mask(distorted_image, output_file_path)
                                else:
                                    save_img(distorted_image, output_file_path, use_JPEG=use_JPEG)

                            except Exception as e:
                                print(f"Error processing {original_file_path} with distortion {distortion_name}: {e}")
                                continue

            print(f"Finished creating dataset for distortion: {distortion_name} with params: {params_str}\n")


###########################################################
#   CONFIG LOADING
###########################################################

def load_config(config_path):
    """Load configuration from a YAML file.
    
    Parameters:
    - config_path: str, path to the YAML configuration file.
    
    Returns:
    - dict with keys 'dataset', 'save_options', 'distortions'
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


###########################################################
#   MAIN METHOD FOR TESTING & DEMONSTRATION PURPOSES
###########################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate distorted images for an arbitrary dataset using a config file."
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    print("This script generates distorted images for an arbitrary dataset.")
    print(f"Loading configuration from: {args.config}")
    
    # Load configuration
    config = load_config(args.config)

    # Extract configuration values
    dataset_config = config.get('dataset', {})
    save_options = config.get('save_options', {})
    distortions_config = config.get('distortions', {})
    
    # Dataset parameters
    csroot = dataset_config.get('root_path')
    output_root = dataset_config.get('output_root')
    dataset_name = dataset_config.get('dataset_name')
    image_subdir = dataset_config.get('image_subdirs')
    mean_color = dataset_config.get('mean_color')
    extensions = tuple(dataset_config.get('image_extensions', ['.png', '.jpg', '.jpeg']))
    is_mask = dataset_config.get('is_mask', False)
    
    # Save options
    use_JPEG = save_options.get('use_jpeg', False)
    
    # Validate required parameters
    if not csroot or not output_root:
        print("Error: 'root_path' and 'output_root' are required in the config file.")
        exit(1)
    
    if not distortions_config:
        print("Error: 'distortions' configuration is empty or missing.")
        exit(1)

    # Call the main function to create distorted datasets
    create_distorted_cityscapes(csroot, output_root,
                                distortions_config=distortions_config,
                                use_JPEG=use_JPEG,
                                mean_color=mean_color,
                                dataset_name=dataset_name,
                                image_subdir=image_subdir,
                                extensions=extensions,
                                is_mask=is_mask)

    print("All distorted datasets have been created successfully.")
