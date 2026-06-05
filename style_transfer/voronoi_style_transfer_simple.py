import torch
from PIL import Image
import numpy as np
from pathlib import Path
import random
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
from stylize import style_transfer, input_transform
import net
from labels import trainId2label
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from PIL import Image, ImageDraw
import argparse

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def generate_voronoi_mask(height, width, num_points):
    # Generate random points
    points = np.random.rand(num_points, 2) * [width, height]
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    # Initial mask creation
    mask = np.zeros((height, width), dtype=np.int32)
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    # Draw initial regions
    for i, region in enumerate(regions):
        polygon = vertices[region]
        polygon = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(polygon, outline=i+1, fill=i+1)
    
    mask = np.array(img)
    
    # Fill gaps using distance to original centers
    if np.any(mask == 0):
        # Get coordinates of unfilled pixels
        zero_pixels = np.where(mask == 0)
        unfilled_coords = np.column_stack([zero_pixels[1], zero_pixels[0]])  # x,y coordinates
        
        # Calculate distances from each unfilled pixel to all centers
        distances = np.zeros((len(unfilled_coords), len(points)))
        for i, center in enumerate(points):
            distances[:, i] = np.sqrt(
                (unfilled_coords[:, 0] - center[0])**2 + 
                (unfilled_coords[:, 1] - center[1])**2
            )
        
        # Assign each unfilled pixel to nearest center
        nearest_center = np.argmin(distances, axis=1) + 1  # +1 because regions are 1-indexed
        mask[zero_pixels] = nearest_center
    
    return mask

def find_images(content_dir, patterns=['*.jpg', '*.jpeg', '*.png']):
    """
    Finds all images in the content directory.
    Returns a list of image paths.
    """
    image_paths = []
    for pattern in patterns:
        image_paths.extend(list(Path(content_dir).glob(pattern)))
    
    return sorted(image_paths)

def stylize(content, style, vgg, decoder, alpha, device):
    """
    Perform style transfer on a single panoptic segment.
    """
    with torch.no_grad():
        content = content.to(device).unsqueeze(0)
        style = style.to(device).unsqueeze(0)
        output = style_transfer(vgg, decoder, content, style, alpha)
    return output 

def main():
    parser = argparse.ArgumentParser(description="Masked Style Transfer with Voronoi Regions")
    parser.add_argument('--num_points', type=int, default=50,
                        help='Number of points for Voronoi diagram')
    parser.add_argument('--output_dirs', nargs='+', 
                        help='Output directories for results')
    parser.add_argument('--content_dirs', nargs='+', 
                        default=['/home/hamscher/datasets/Ranus_splits/RGB/train'],
                        help='Directories containing content images')
    parser.add_argument('--style_dir', type=str, 
                        default='~/datasets/train',
                        help='Directory containing style images')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Style transfer strength parameter')
    parser.add_argument('--content_size', type=int, default=0,
                        help='Size for content images (0 means original size)')
    parser.add_argument('--style_size', type=int, default=512,
                        help='Size for style images')
    parser.add_argument('--crop', type=int, default=0,
                        help='Crop size for content images')
    parser.add_argument('--stylize_proportion', type=float, default=1.0,
                        help='Proportion of regions to stylize (0.0-1.0)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Convert style images to grayscale before stylization')
    parser.add_argument('--save_styles', action='store_true',
                        help='Save style images used for each region')
    parser.add_argument('--content_file', type=str, default=None,
                        help='Path to a single content image to process')
    args = parser.parse_args() 

    # Get arguments with defaults applied
    content_dirs = [Path(p).expanduser() for p in args.content_dirs]
    style_dir = Path(args.style_dir).expanduser()
    output_dirs = args.output_dirs if args.output_dirs else [f"stylized_output_{args.num_points}_regions" for _ in args.content_dirs]
    output_dirs = [Path(p).expanduser() for p in output_dirs]
    alpha = args.alpha
    content_size = args.content_size
    style_size = args.style_size
    crop = args.crop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stylize_proportion = args.stylize_proportion
    num_points = args.num_points

    for content_dir, output_dir in zip(content_dirs, output_dirs):
        print(f"Stylizing Images in {content_dir}.")
        content_dir = Path(content_dir).resolve()
        style_dir = Path(style_dir).resolve()
        output_dir = Path(output_dir).resolve()

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare model
        decoder = net.decoder
        vgg = net.vgg

        decoder.eval()
        vgg.eval()

        decoder.load_state_dict(torch.load('models/decoder.pth'))
        vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
        vgg = nn.Sequential(*list(vgg.children())[:31])

        vgg.to(device)
        decoder.to(device)

        # Find images
        if args.content_file:
            content_paths = [Path(args.content_file).expanduser()]
            print(f'Processing single image: {content_paths[0]}')
        else:
            content_paths = find_images(content_dir)
            print(f'Found {len(content_paths)} images')

        # Collect style files
        extensions = ['png', 'jpeg', 'jpg']
        styles = []
        for ext in extensions:
            styles.extend(list(style_dir.rglob(f'*.{ext}')))

        assert len(styles) > 0, f'No images with specified extensions found in style directory {style_dir}'
        styles = sorted(styles)
        print(f'Found {len(styles)} style images in {style_dir}')

        # Disable decompression bomb errors
        Image.MAX_IMAGE_PIXELS = None
        skipped_imgs = []

        # Define transforms
        content_tf = input_transform(content_size, crop)
        style_tf = input_transform(style_size, 0)
        
        # Process images
        with tqdm(total=len(content_paths)) as pbar:
            for content_path in content_paths:    
                try:
                    # Load content image
                    content_img = Image.open(content_path).convert('RGB')
                    content_name = content_path.stem
                    content_tensor = content_tf(content_img).to(device)
                    
                    # Get correct dimensions from the tensor
                    _, h, w = content_tensor.shape
                    
                    # Generate Voronoi mask with correct dimensions
                    voronoi_mask = generate_voronoi_mask(w, h, num_points=num_points)
                    
                    # Make sure voronoi_mask matches tensor dimensions
                    voronoi_mask = torch.from_numpy(voronoi_mask).float()
                    if voronoi_mask.shape != (h, w):
                        voronoi_mask = voronoi_mask.transpose(0, 1)
                    
                    # Initialize result tensor
                    result = torch.zeros_like(content_tensor).unsqueeze(0).to(device)
                    new_style_mask = torch.zeros_like(voronoi_mask)

                    # Process each unique ID
                    unique_ids = torch.unique(voronoi_mask)
                    for idx in unique_ids:
                        if idx == 0:
                            continue
                        
                        if not stylize_proportion == 1.0 and random.random() > stylize_proportion:
                            style_class = 255
                            new_style_mask[voronoi_mask == idx] = style_class
                            continue

                        # Select random style
                        style_path = random.choice(styles)

                        # Load and preprocess style image
                        if not args.grayscale:
                            style_img = Image.open(style_path).convert('RGB')
                        else:
                            style_img = Image.open(style_path).convert('L')
                            style_img = style_img.convert('RGB')
                        
                        if args.save_styles:
                            style_save_dir = output_dir / f"{content_name}_styles"
                            style_save_dir.mkdir(parents=True, exist_ok=True)
                            # Save original style image
                            orig_style = Image.open(style_path).convert('RGB')
                            orig_style.save(style_save_dir / f"region_{idx.item():.0f}_style_orig.png")
                            # Save grayscaled version if requested
                            if args.grayscale:
                                style_img.save(style_save_dir / f"region_{idx.item():.0f}_style_gray.png")

                        style_tensor = style_tf(style_img)

                        # Perform style transfer
                        stylized_output = stylize(content_tensor, style_tensor, vgg, decoder, alpha, device)

                        # Create and apply mask
                        id_mask = (voronoi_mask == idx).float()
                        mask_3d = id_mask.repeat(3, 1, 1).unsqueeze(0).to(device)
                        result += mask_3d * stylized_output

                        # Assign a random class ID between 0 and 18 (Cityscapes classes)
                        style_class = random.randint(0, 18)
                        new_style_mask[voronoi_mask.numpy() == idx.item()] = style_class

                    # Add unstylized regions back
                    id_mask = (new_style_mask == 255).float()
                    mask_3d = id_mask.repeat(3, 1, 1).unsqueeze(0).to(device)
                    result += mask_3d * content_tensor.unsqueeze(0)

                    # Save stylized image
                    out_filename = f"{content_name}{content_path.suffix}"
                    output_name = output_dir / out_filename
                    save_image(result.cpu(), output_name, padding=0)

                    # Save style mask
                    new_mask_name = f"{content_name}_style_mask.png"
                    new_mask_path = output_dir / new_mask_name
                    Image.fromarray(new_style_mask.numpy().astype(np.uint8)).save(new_mask_path)

                    # Create and save RGB style mask
                    rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
                    for train_id in trainId2label.keys():
                        rgb_color = trainId2label[train_id].color
                        mask = (new_style_mask == train_id)
                        rgb_array[mask] = rgb_color

                    rgb_image = Image.fromarray(rgb_array)
                    rgb_image_name = f"{content_name}_style_mask_rgb.png"
                    rgb_image_path = output_dir / rgb_image_name
                    rgb_image.save(rgb_image_path)

                except Exception as e:
                    print(f'Skipping stylization of {content_path} due to an error: {str(e)}')
                    skipped_imgs.append(content_path)
                    continue
                finally:
                    pbar.update(1)

        # Save skipped images list
        if skipped_imgs:
            skipped_file = output_dir / 'skipped_imgs.txt'
            skipped_file.parent.mkdir(parents=True, exist_ok=True)
            with open(skipped_file, 'w') as f:
                for item in skipped_imgs:
                    f.write(f"{str(item)}\n")

if __name__ == '__main__':
    main()
