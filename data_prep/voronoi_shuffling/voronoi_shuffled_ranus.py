
""""
General setup for voronoi diagram
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.voronoi_plot_2d.html
Generate polygones from voronoi cells
https://gist.github.com/pv/8036995

Draw and fill voronoi diagrams
"""
import argparse
import random
from dataloader import RANUS
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import translate

from torchvision import transforms

# import cv2
from pathlib import Path
import os
from PIL import Image, ImageDraw
from tqdm import tqdm

from torchvision import datasets

# # Ensure polygon points are within bounds
# def clamp_point(point, width, height):
#     x, y = point
#     return max(0, min(x, width - 1)), max(0, min(y, height - 1))


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Subclass of ImageFolder that returns (image, label, image_path) tuple.
    """
    def __getitem__(self, index):
        # Standard behavior: get (image, label)
        original_tuple = super().__getitem__(index)
        # Get the image path
        path = self.imgs[index][0]
        # Return image, label, and path
        return (*original_tuple, path)




def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Limit infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices),


def randomize_voronoi_diagram(num_points, dim, name, img_to_shuffle, label_to_shuffle, split, rng, result_dir="results", size=(2048,1024), post_transform=None, dummy_shift=100, process_labels=True, process_images=True, image_subfolder="RGB", label_subfolder="PseudoGT" ):
    '''
    Generate voronoi centers and fill each cell with a texture image of a certain class

    Parameters
    ----------
    num_points: Number of voronoi centers
    dim: dimension of the diagram (e.g. 2D)
    name: name to identify the generated diagram
    process_labels: Whether to process and save labels (False when processing NIR with existing labels)
    process_images: Whether to process and save images (False when processing labels only)
    image_subfolder: Subfolder name for images (e.g., 'RGB' or 'NIR')
    label_subfolder: Subfolder name for labels (e.g., 'PseudoGT' or other label types)

    Returns
    ---------
    Voronoi Diagram
    '''
    fig, ax1 = plt.subplots(dpi=300)
    points = rng.random((num_points, dim))*size

    # add 4 distant dummy points
    # points = np.append(points, [[2440, 1440], [-0, 1440], [2440, -0], [-0, -0]], axis=0)
    # points = np.append(points, [[size[0] + 150, size[1] + 150 ], [-0, size[1] + 150], [size[0] + 150, -0], [-0, -0]], axis=0)
    points = np.append(points, [[size[0] + dummy_shift, size[1] + dummy_shift ], [-0, size[1] + dummy_shift], [size[0] + dummy_shift, -0], [-0, -0]], axis=0)
    vor = Voronoi(points)

    voronoi_plot_2d(vor, ax1, show_vertices=False)
    plt.xlim([0, size[0]]), plt.ylim([0, size[1]])

    # generate Voronoi with finite cells
    regions, vertices = voronoi_finite_polygons_2d(vor)

    voronoi_img = Image.new('RGB', size, 0)
    voronoi_label_img = Image.new('L', size, 0)
    
    
    for region in regions:
        if region:  # valid region (not empty polygonpoint list) and not -1 in region:
            polygon = vertices[region]

            voronoi_img, voronoi_label = cropping(voronoi_img,
                                                  voronoi_label_img,
                                                  img_to_shuffle,
                                                  label_to_shuffle,
                                                  polygon,
                                                  size)

    if process_images:
        os.makedirs(os.path.join(result_dir, image_subfolder, split), exist_ok=True)
    if process_labels:
        os.makedirs(os.path.join(result_dir, label_subfolder, split), exist_ok=True)
    
    if post_transform:
        trasformed_voronoi_img = post_transform(voronoi_img)
        trasformed_voronoi_label = post_transform(voronoi_label)
    else: 
        trasformed_voronoi_img = voronoi_img
        trasformed_voronoi_label = voronoi_label
    
    if process_images:
        trasformed_voronoi_img.save(os.path.join(result_dir, image_subfolder, split, str(name)))
    if process_labels:
        trasformed_voronoi_label.save(os.path.join(result_dir, label_subfolder, split, str(name)))
    # plt.gca().invert_yaxis()
    # ax1.plot(points[:-4, 0], points[:-4, 1], 'ro', markersize=2, zorder=10)

    # fig.savefig(f"Images/voronoi_structure_{name}.png")
    plt.close()


def cropping(voronoi, voronoi_label, img_to_shuffle, label_to_shuffle, polygon, size):
    '''
    Fill a single Voronoi cell with texture
    '''

    width, height = size
    
    # Define the bounding box
    bounding_box = box(0, 0, width, height)
    polygon = Polygon(polygon)

    # Crop the polygon using intersection
    cropped_polygon_points = polygon.intersection(bounding_box)
    if cropped_polygon_points.area == 0.0:
        return voronoi, voronoi_label

    # calculate random shift
    shift = (np.int32(random.uniform(-cropped_polygon_points.bounds[0], size[0] - cropped_polygon_points.bounds[2])),
             np.int32(random.uniform(-cropped_polygon_points.bounds[1], size[1] - cropped_polygon_points.bounds[3])))
    
    shifted_polygon = translate(cropped_polygon_points, xoff=shift[0], yoff=shift[1])

    # polygon_points = np.array([clamp_point(p, width, height) for p in polygon])


    polygon_filled = Image.new('RGB', size, 0)
    polygon_label_filled = Image.new('L', size, 0)
    polygon_mask_shifted = Image.new('L', size, 0)

    polygon_mask = Image.new('L', size, 0)
    polygon_path_shifted = list(shifted_polygon.exterior.coords)
    polygon_path =list(cropped_polygon_points.exterior.coords)
    ImageDraw.Draw(polygon_mask_shifted).polygon(polygon_path_shifted, fill=(255), outline=(255))
    ImageDraw.Draw(polygon_mask).polygon(polygon_path, fill=(255), outline=(255))

    #fill voronoi diagram
    polygon_filled.paste(img_to_shuffle, (0,0), polygon_mask_shifted)
    voronoi.paste(polygon_filled, (-shift[0], -shift[1]), polygon_mask_shifted)
    polygon_label_filled.paste(label_to_shuffle, (0,0), polygon_mask_shifted)
    voronoi_label.paste(polygon_label_filled, (-shift[0], -shift[1]), polygon_mask_shifted)

    return voronoi, voronoi_label




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Voronoi shuffled RANUS dataset')
    parser.add_argument('--cell_number', type=int, default=128, help='Number of Voronoi cells')
    parser.add_argument('--img_root', type=str, default="../../data/Ranus_prepared/NIR/",
                        help='Root path to RANUS images')
    parser.add_argument('--label_root', type=str, default="../../data/Ranus_prepared/GT_8class/",
                        help='Root path to RANUS labels')
    parser.add_argument('--splits', type=str, nargs='+', default=['val'],
                        help='Splits to process (e.g., train val test)')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Output directory (default: Voronoi_RANUS_<cell_number>)')
    parser.add_argument('--seed', type=int, default=4224, help='Random seed')
    parser.add_argument('--images_only', action='store_true',
                        help='Process only images, skip labels (use when processing NIR with existing labels)')
    parser.add_argument('--labels_only', action='store_true',
                        help='Process only labels, skip images (use when reusing existing shuffled images)')
    parser.add_argument('--image_subfolder', type=str, default='RGB',
                        help='Subfolder name for output images (e.g., "RGB" or "NIR")')
    parser.add_argument('--label_subfolder', type=str, default='PseudoGT',
                        help='Subfolder name for output labels (e.g., "PseudoGT" or other label types)')
    
    args = parser.parse_args()
    
    cell_number = args.cell_number
    dimensionen = 2
    size = (1024, 1024)  # RANUS image size
    img_root = args.img_root
    label_root = args.label_root
    result_dir = args.result_dir if args.result_dir else Path(img_root).parent / f"Voronoi_shuffled_{cell_number}"
    
    rng = np.random.default_rng(args.seed)  # random number generator for random handling in numpy
    post_transformation = None
    
    # Process each split
    for split in args.splits:
        print(f"\nProcessing {split} split...")
        
        dataset_to_shuffle = RANUS(img_root=img_root,
                                   label_root=label_root, 
                                   split=split
                                   )
        
        for img_to_shuffle, label_to_shuffle, path in tqdm(dataset_to_shuffle, desc=f"{split} split"):
            randomize_voronoi_diagram(cell_number,
                                      dimensionen,
                                      name=os.path.basename(path),
                                      split=split,
                                      rng=rng,
                                      img_to_shuffle=img_to_shuffle,
                                      label_to_shuffle=label_to_shuffle,
                                      result_dir=result_dir,
                                      size=size,
                                      post_transform=post_transformation,
                                      process_labels=not args.images_only,
                                      process_images=not args.labels_only,
                                      image_subfolder=args.image_subfolder,
                                      label_subfolder=args.label_subfolder
                                      )
    
    print(f"\nCompleted! Output saved to: {result_dir}")
# end main
