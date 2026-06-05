import os
from collections import namedtuple
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


from PIL import Image


class Cityscapes(Dataset):
    """`
    Cityscapes Dataset http://www.cityscapes-dataset.com/
    Labels based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled',            0,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4,  255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5,  255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6,  255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7,  0,   'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8,  1,   'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9,  255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2,   'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3,   'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4,   'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5,   'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6,   'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7,   'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8,   'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9,   'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10,  'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11,  'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12,  'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13,  'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14,  'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15,  'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16,  'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17,  'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18,  'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, -1,  'vehicle', 7, False, True, (0, 0, 142)),
    ]

    """Normalization parameters"""
    # ImageNet normalization
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)

    # Cityscapes training data normalization (c)
    # mean = (0.2868955, 0.3251328, 0.2838913)
    # std = (0.1761364, 0.1809918, 0.1777224)

    # grayscale (c)
    # mean = (0.3090155)
    # std = (0.1786242)

    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)
    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))


    def __init__(self,
                 img_root: str,
                 label_root: str,
                 split: str,
                 mode: str = "fine",
                 target_type: str = "labelTrainIds",
                 ignore_cities: list = [],
                 switch_to_train_id: bool = True
                ) -> None:
        """
        Cityscapes dataset loader
        """
        if label_root != img_root:
            print("Are you using the original Cityscapes data and structure? Please verify you are using the right data. Will exit now...")
            exit()
        self.root = img_root
        self.target_root = label_root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.switch_to_train_id = switch_to_train_id
        self.gray = False

        # data root
        self.images_dir = os.path.join(self.root, 'leftImg8bit', self.split.split('_')[0])
        self.targets_dir = os.path.join(self.root, self.mode, self.split.split('_')[0])

        self.images = []
        self.targets = []

        for city in os.listdir(self.images_dir):
            if (self.split == 'train' or self.split == 'val') and city not in ignore_cities:
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)
                for file_name in os.listdir(img_dir):
                    target_name = f'{file_name.split("_leftImg8bit")[0]}_{self.mode}_{target_type}.png'
                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(os.path.join(target_dir, target_name))

                    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.gray:
            image = Image.open(self.images[index]).convert('L')
        else:
            image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        np_label = np.uint8(np.array(target))
        if self.switch_to_train_id:
            # transform segmentation mask according to train_ids
            mask = 255*np.ones((np_label.shape), dtype=np.uint8)
            for i in np.unique(np_label):
                mask[np_label == i] = self.labels[i].train_id
            target = Image.fromarray(mask)

        path = self.images[index]

        return image, target, path

    def __len__(self) -> int:
        return len(self.images)


class GTA5(Dataset):
    """`
    GTA5 Dataset for semantic segmentation
    Uses the same 19 classes as Cityscapes (trainIds)
    """
    
    # Reuse Cityscapes labels for compatibility (19 classes)
    labels = Cityscapes.labels
    ignore_in_eval_ids = Cityscapes.ignore_in_eval_ids
    label_ids = Cityscapes.label_ids
    train_ids = Cityscapes.train_ids
    train_id2id = Cityscapes.train_id2id
    num_label_ids = Cityscapes.num_label_ids
    num_train_ids = Cityscapes.num_train_ids
    id2label = Cityscapes.id2label
    train_id2label = Cityscapes.train_id2label
    color_palette_train_ids = Cityscapes.color_palette_train_ids

    def __init__(self,
                 img_root: str,
                 label_root: str,
                 split: str
                ) -> None:
        """
        Args:
            img_root: Root directory containing images/<split>/ folder
            label_root: Root directory containing labels_trainid/<split>/ folder
            split: Dataset split ('train', 'val', or 'test')
        """
        self.img_root = img_root
        self.label_root = label_root
        self.split = split
        
        # Get image and label paths
        img_dir = os.path.join(img_root, split)
        label_dir = os.path.join(label_root, split)
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
        
        # Get all image files (assuming .png format)
        self.images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                             if f.endswith('.png')])
        self.labels = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) 
                             if f.endswith('.png')])
        
        assert len(self.images) == len(self.labels), \
            f"Number of images ({len(self.images)}) != number of labels ({len(self.labels)})"
        
        print(f"GTA5 {split}: Found {len(self.images)} images")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            tuple: (image, target, path) where target is the segmentation mask
        """
        img_path = self.images[index]
        label_path = self.labels[index]
        
        image = Image.open(img_path).convert('RGB')
        target = Image.open(label_path)
        
        return image, target, img_path

    def __len__(self) -> int:
        return len(self.images)


class RANUS(Dataset):
    """`
    RANUS Dataset for semantic segmentation
    Uses the same 19 classes as Cityscapes (trainIds) via PseudoGT labels
    """
    
    # Reuse Cityscapes labels for compatibility (19 classes)
    labels = Cityscapes.labels
    ignore_in_eval_ids = Cityscapes.ignore_in_eval_ids
    label_ids = Cityscapes.label_ids
    train_ids = Cityscapes.train_ids
    train_id2id = Cityscapes.train_id2id
    num_label_ids = Cityscapes.num_label_ids
    num_train_ids = Cityscapes.num_train_ids
    id2label = Cityscapes.id2label
    train_id2label = Cityscapes.train_id2label
    color_palette_train_ids = Cityscapes.color_palette_train_ids

    def __init__(self,
                 img_root: str,
                 label_root: str,
                 split: str
                ) -> None:
        """
        Args:
            img_root: Root directory containing RGB/<split>/ folder
            label_root: Root directory containing PseudoGT/<split>/ folder
            split: Dataset split ('train', 'val', or 'test')
        """
        self.img_root = img_root
        self.label_root = label_root
        self.split = split
        
        # Get image and label paths
        img_dir = os.path.join(img_root, split)
        label_dir = os.path.join(label_root, split)
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
        
        # Get all image files (assuming .png format)
        self.images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                             if f.endswith('.png')])
        self.labels = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) 
                             if f.endswith('.png')])
        
        assert len(self.images) == len(self.labels), \
            f"Number of images ({len(self.images)}) != number of labels ({len(self.labels)})"
        
        print(f"RANUS {split}: Found {len(self.images)} images")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            tuple: (image, target, path) where target is the segmentation mask
        """
        img_path = self.images[index]
        label_path = self.labels[index]
        
        image = Image.open(img_path).convert('RGB')
        target = Image.open(label_path)
        
        return image, target, img_path

    def __len__(self) -> int:
        return len(self.images)


class PascalContext33classes(Dataset):
    PascalContextClass = namedtuple('PascalContextClass', ['name',
                                                           'id', 
                                                           'train_id',
                                                           'color'
                                                           ])
    labels = [
        PascalContextClass('void',          0,  255,(  0,  0,  0)),
        PascalContextClass('aeroplane',     2,    0,(128,  0,  0)),
        PascalContextClass('bicycle',      23,    1,(  0,128,  0)),
        PascalContextClass('bird',         25,    2,(128,128,  0)),
        PascalContextClass('boat',         31,    3,(  0,  0,128) ),
        PascalContextClass('bottle',       34,    4,(128,  0,128)),
        PascalContextClass('bus',          45,    5,(  0,128,128)),
        PascalContextClass('car',          59,    6,(128,128,128)),
        PascalContextClass('cat',          65,    7,( 64,  0,  0)),
        PascalContextClass('chair',        72,    8,(192,  0,  0)),
        PascalContextClass('cow',          98,    9,( 64,128,  0)),
        PascalContextClass('diningtabel', 397,   10,(192,128,  0)),
        PascalContextClass('dog',         113,   11,( 64,  0,128)),
        PascalContextClass('horse',       207,   12,(192,  0,128)),
        PascalContextClass('motorbike',   258,   13,( 64,128,128)),
        PascalContextClass('person',      284,   14,(192,128,128)),
        PascalContextClass('pottedplant', 308,   15,(  0, 64,  0)),
        PascalContextClass('sheep',       347,   16,(128, 64,  0)),
        PascalContextClass('sofa',        368,   17,(  0,192,  0)),
        PascalContextClass('train',       416,   18,(128,192,  0)),
        PascalContextClass('tvmonitor',   427,   19,(  0, 64,128)),
        PascalContextClass('sky',         360,   20,(  0,192, 64)),
        PascalContextClass('grass',       187,   21,(  0,  0,192)),
        PascalContextClass('ground',      189,   22,(128,  0,192)),
        PascalContextClass('road',        324,   23,( 64,128,192)),
        PascalContextClass('building',     44,   24,(192, 64,  0)),
        PascalContextClass('tree',        420,   25,( 64,192,  0)),
        PascalContextClass('water',       445,   26,(192, 64, 64)),
        PascalContextClass('mountain',    259,   27,( 64,  0, 64)),
        PascalContextClass('wall',        440,   28,(128, 64,128)),
        PascalContextClass('floor',       158,   29,(128,  0, 64)),
        PascalContextClass('track',       415,   30,(128, 64,192)),
        PascalContextClass('keyboard',    220,   31,(  0,128,192)),
        PascalContextClass('ceiling',      68,   32,(192,192,  0))
    ]
    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
  
    # for i in range(len(labels)):
    #     if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
    #         ignore_in_eval_ids.append(labels[i].train_id)

    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        #if labels[i].train_id not in ignore_in_eval_ids:
        train_ids.append(labels[i].train_id)
        color_palette_train_ids[labels[i].train_id] = labels[i].color
        train_id2id.append(labels[i].id)
       
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_2label = {label.color: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))


    # mean: [  ]
    # std: [  ] 

    def __init__(self,
                    img_root: str,
                    label_root: str,
                    split: str,
                    switch_to_train_id: bool = False
                    ) -> None:
        """
        Pascal Context dataset loader
        """
        self.root = img_root
        self.target_root = label_root
        self.split = split
        self.switch_to_train_id = switch_to_train_id
        # build transformation


        # data root paths
        if self.split == 'val_train':
            exit("Please use the val split since a seperate test split exists")
            self.split = 'train'
        self.images_dir = os.path.join(self.root, self.split)
        self.targets_dir = os.path.join(self.target_root, self.split)

        self.images = []
        self.targets = []

        for root, dirs, files in os.walk(self.targets_dir):
            for file in files:
                img_name = f'{file.split(".")[0]}.jpg'
                
                self.images.append(os.path.join(self.images_dir, img_name))
                self.targets.append(os.path.join(self.targets_dir, file))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        path = self.images[index]

        np_label = np.uint8(np.array(target))
        if self.switch_to_train_id:
            # transform segmentation mask according to train_ids
            mask = 255*np.ones((np_label.shape), dtype=np.uint8)
            for i in np.unique(np_label):
                mask[np_label == i] = self.labels[i].train_id
            target = Image.fromarray(mask)
        
        return image, target, path

    def __len__(self) -> int:
        return len(self.images)





class ADE20k(Dataset):
    ADE20kClass = namedtuple('ADE20kClass', ['name',
                                             'id', 
                                             'train_id',
                                             'color'
                                             ])
    labels = [
        ADE20kClass(    'background',            0,     255,        (120, 120, 120)),
        ADE20kClass(    'wall',                  1,     0,        (120, 120, 120)),
        ADE20kClass(     'building',             2,     1,        (180, 120, 120)),
        ADE20kClass(     'sky',                  3,     2,        (6, 230, 230)),
        ADE20kClass(     'floor',                4,     3,        (80, 50, 50)),
        ADE20kClass(     'tree',                 5,     4,        (4, 200, 3)),
        ADE20kClass(     'ceiling',              6,     5,           (120, 120, 80)),
        ADE20kClass(     'road',                 7,     6,        (140, 140, 140)),
        ADE20kClass(     'bed ',                 8,     7,        (204, 5, 255)),
        ADE20kClass(     'windowpane',           9,     8,              (230, 230, 230)),
        ADE20kClass(     'grass',                10,    9,         (4, 250, 7)),
        ADE20kClass(     'cabinet',              11,    10,           (224, 5, 255)),
        ADE20kClass(     'sidewalk',             12,    11,            (235, 255, 7)),
        ADE20kClass(     'person',               13,    12,          (150, 5, 61)),
        ADE20kClass(     'earth',                14,    13,         (120, 120, 70)),
        ADE20kClass(     'door',                 15,    14,        (8, 255, 51)),
        ADE20kClass(     'table',                16,    15,         (255, 6, 82)),
        ADE20kClass(     'mountain',             17,    16,            (143, 255, 140)),
        ADE20kClass(     'plant',                18,    17,         (204, 255, 4)),
        ADE20kClass(     'curtain',              19,    18,           (255, 51, 7)),
        ADE20kClass(     'chair',                20,    19,         (204, 70, 3)),
        ADE20kClass(     'car',                  21,    20,       (0, 102, 200)),
        ADE20kClass(     'water',                22,    21,         (61, 230, 250)),
        ADE20kClass(     'painting',             23,    22,            (255, 6, 51)),
        ADE20kClass(     'sofa',                 24,    23,        (11, 102, 255)),
        ADE20kClass(     'shelf',                25,    24,         (255, 7, 71)),
        ADE20kClass(     'house',                26,    25,         (255, 9, 224)),
        ADE20kClass(     'sea',                  27,    26,       (9, 7, 230)),
        ADE20kClass(     'mirror',               28,    27,          (220, 220, 220)),
        ADE20kClass(     'rug',                  29,    28,       (255, 9, 92)),
        ADE20kClass(     'field',                30,    29,         (112, 9, 255)),
        ADE20kClass(     'armchair',             31,    30,            (8, 255, 214)),
        ADE20kClass(     'seat',                 32,    31,        (7, 255, 224)),
        ADE20kClass(     'fence',                33,    32,         (255, 184, 6)),
        ADE20kClass(     'desk',                 34,    33,        (10, 255, 71)),
        ADE20kClass(     'rock',                 35,    34,        (255, 41, 10)),
        ADE20kClass(     'wardrobe',             36,    35,            (7, 255, 255)),
        ADE20kClass(     'lamp',                 37,    36,        (224, 255, 8)),
        ADE20kClass(     'bathtub',              38,    37,           (102, 8, 255)),
        ADE20kClass(     'railing',              39,    38,           (255, 61, 6)),
        ADE20kClass(     'cushion',              40,    39,           (255, 194, 7)),
        ADE20kClass(     'base',                 41,    40,        (255, 122, 8)),
        ADE20kClass(     'box',                  42,    41,       (0, 255, 20)),
        ADE20kClass(     'column',               43,    42,          (255, 8, 41)),
        ADE20kClass(     'signboard',            44,    43,             (255, 5, 153)),
        ADE20kClass(     'chest of drawers',     45,    44,                    (6, 51, 255)),
        ADE20kClass(     'counter',              46,    45,           (235, 12, 255)),
        ADE20kClass(     'sand',                 47,    46,        (160, 150, 20)),
        ADE20kClass(     'sink',                 48,    47,        (0, 163, 255)),
        ADE20kClass(     'skyscraper',           49,    48,              (140, 140, 140)),
        ADE20kClass(     'fireplace',            50,    49,             (250, 10, 15)),
        ADE20kClass(     'refrigerator',         51,    50,                (20, 255, 0)),
        ADE20kClass(     'grandstand',           52,    51,              (31, 255, 0)),
        ADE20kClass(     'path',                 53,    52,        (255, 31, 0)),
        ADE20kClass(     'stairs',               54,    53,          (255, 224, 0)),
        ADE20kClass(     'runway',               55,    54,          (153, 255, 0)),
        ADE20kClass(     'case',                 56,    55,        (0, 0, 255)),
        ADE20kClass(     'pool table',           57,    56,              (255, 71, 0)),
        ADE20kClass(     'pillow',               58,    57,          (0, 235, 255)),
        ADE20kClass(     'screen door',          59,    58,               (0, 173, 255)),
        ADE20kClass(     'stairway',             60,    59,            (31, 0, 255)),
        ADE20kClass(     'river',                61,    60,         (11, 200, 200)),
        ADE20kClass(     'bridge',               62,    61,          (255, 82, 0)),
        ADE20kClass(     'bookcase',             63,    62,            (0, 255, 245)),
        ADE20kClass(     'blind',                64,    63,         (0, 61, 255)),
        ADE20kClass(     'coffee table',         65,    64,                (0, 255, 112)),
        ADE20kClass(     'toilet',               66,    65,          (0, 255, 133)),
        ADE20kClass(     'flower',               67,    66,          (255, 0, 0)),
        ADE20kClass(     'book',                 68,    67,        (255, 163, 0)),
        ADE20kClass(     'hill',                 69,    68,        (255, 102, 0)),
        ADE20kClass(     'bench',                70,    69,         (194, 255, 0)),
        ADE20kClass(     'countertop',           71,    70,              (0, 143, 255)),
        ADE20kClass(     'stove',                72,    71,         (51, 255, 0)),
        ADE20kClass(     'palm',                 73,    72,        (0, 82, 255)),
        ADE20kClass(     'kitchen island',       74,    73,                  (0, 255, 41)),
        ADE20kClass(     'computer',             75,    74,            (0, 255, 173)),
        ADE20kClass(     'swivel chair',         76,    75,                (10, 0, 255)),
        ADE20kClass(     'boat',                 77,    76,        (173, 255, 0)),
        ADE20kClass(     'bar',                  78,    77,       (0, 255, 153)),
        ADE20kClass(     'arcade machine',       79,    78,                  (255, 92, 0)),
        ADE20kClass(     'hovel',                80,    79,         (255, 0, 255)),
        ADE20kClass(     'bus',                  81,    80,       (255, 0, 245)),
        ADE20kClass(     'towel',                82,    81,         (255, 0, 102)),
        ADE20kClass(     'light',                83,    82,         (255, 173, 0)),
        ADE20kClass(     'truck',                84,    83,         (255, 0, 20)),
        ADE20kClass(     'tower',                85,    84,         (255, 184, 184)),
        ADE20kClass(     'chandelier',           86,    85,              (0, 31, 255)),
        ADE20kClass(     'awning',               87,    86,          (0, 255, 61)),
        ADE20kClass(     'streetlight',          88,    87,               (0, 71, 255)),
        ADE20kClass(     'booth',                89,    88,         (255, 0, 204)),
        ADE20kClass(     'television receiver',  90,    89,           (0, 255, 194)),
        ADE20kClass(     'airplane',             91,    90,            (0, 255, 82)),
        ADE20kClass(     'dirt track',           92,    91,              (0, 10, 255)),
        ADE20kClass(     'apparel',              93,    92,           (0, 112, 255)),
        ADE20kClass(     'pole',                 94,    93,        (51, 0, 255)),
        ADE20kClass(     'land',                 95,    94,        (0, 194, 255)),
        ADE20kClass(     'bannister',            96,    95,             (0, 122, 255)),
        ADE20kClass(     'escalator',            97,    96,             (0, 255, 163)),
        ADE20kClass(     'ottoman',              98,    97,           (255, 153, 0)),
        ADE20kClass(     'bottle',               99,    98,          (0, 255, 10)),
        ADE20kClass(     'buffet',               100,   99,          (255, 112, 0)),
        ADE20kClass(     'poster',               101,   100,          (143, 255, 0)),
        ADE20kClass(     'stage',                102,   101,         (82, 0, 255)),
        ADE20kClass(     'van',                  103,   102,       (163, 255, 0)),
        ADE20kClass(     'ship',                 104,   103,        (255, 235, 0)),
        ADE20kClass(     'fountain',             105,   104,            (8, 184, 170)),
        ADE20kClass(     'conveyer belt',        106,   105,                 (133, 0, 255)),
        ADE20kClass(     'canopy',               107,   106,          (0, 255, 92)),
        ADE20kClass(     'washer',               108,   107,          (184, 0, 255)),
        ADE20kClass(     'plaything',            109,   108,             (255, 0, 31)),
        ADE20kClass(     'swimming pool',        110,   109,                 (0, 184, 255)),
        ADE20kClass(     'stool',                111,   110,         (0, 214, 255)),
        ADE20kClass(     'barrel',               112,   111,          (255, 0, 112)),
        ADE20kClass(     'basket',               113,   112,          (92, 255, 0)),
        ADE20kClass(     'waterfall',            114,   113,             (0, 224, 255)),
        ADE20kClass(     'tent',                 115,   114,        (112, 224, 255)),
        ADE20kClass(     'bag',                  116,   115,       (70, 184, 160)),
        ADE20kClass(     'minibike',             117,   116,            (163, 0, 255)),
        ADE20kClass(     'cradle',               118,   117,          (153, 0, 255)),
        ADE20kClass(     'oven',                 119,   118,        (71, 255, 0)),
        ADE20kClass(     'ball',                 120,   119,        (255, 0, 163)),
        ADE20kClass(     'food',                 121,   120,        (255, 204, 0)),
        ADE20kClass(     'step',                 122,   121,        (255, 0, 143)),
        ADE20kClass(     'tank',                 123,   122,        (0, 255, 235)),
        ADE20kClass(     'trade name',           124,   123,              (133, 255, 0)),
        ADE20kClass(     'microwave',            125,   124,             (255, 0, 235)),
        ADE20kClass(     'pot',                  126,   125,       (245, 0, 255)),
        ADE20kClass(     'animal',               127,   126,          (255, 0, 122)),
        ADE20kClass(     'bicycle',              128,   127,           (255, 245, 0)),
        ADE20kClass(     'lake',                 129,   128,        (10, 190, 212)),
        ADE20kClass(     'dishwasher',           130,   129,              (214, 255, 0)),
        ADE20kClass(     'screen',               131,   130,          (0, 204, 255)),
        ADE20kClass(     'blanket',              132,   131,           (20, 0, 255)),
        ADE20kClass(     'sculpture',            133,   132,             (255, 255, 0)),
        ADE20kClass(     'hood',                 134,   133,        (0, 153, 255)),
        ADE20kClass(     'sconce',               135,   134,          (0, 41, 255)),
        ADE20kClass(     'vase',                 136,   135,        (0, 255, 204)),
        ADE20kClass(     'traffic light',        137,   136,                 (41, 0, 255)),
        ADE20kClass(     'tray',                 138,   137,        (41, 255, 0)),
        ADE20kClass(     'ashcan',               139,   138,          (173, 0, 255)),
        ADE20kClass(     'fan',                  140,   139,       (0, 245, 255)),
        ADE20kClass(     'pier',                 141,   140,        (71, 0, 255)),
        ADE20kClass(     'crt screen',           142,   141,              (122, 0, 255)),
        ADE20kClass(     'plate',                143,   142,         (0, 255, 184)),
        ADE20kClass(     'monitor',              144,   143,           (0, 92, 255)),
        ADE20kClass(     'bulletin board',       145,   144,                  (184, 255, 0)),
        ADE20kClass(     'shower',               146,   145,          (0, 133, 255)),
        ADE20kClass(     'radiator',             147,   146,            (255, 214, 0)),
        ADE20kClass(     'glass',                148,   147,         (25, 194, 194)),
        ADE20kClass(     'clock',                149,  148,         (102, 255, 0)),
        ADE20kClass(     'flag',                 150,  149,     (92, 0, 255))
        ]
      

    
    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
  
    # for i in range(len(labels)):
    #     if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
    #         ignore_in_eval_ids.append(labels[i].train_id)

    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        #if labels[i].train_id not in ignore_in_eval_ids:
        train_ids.append(labels[i].train_id)
        color_palette_train_ids[labels[i].train_id] = labels[i].color
        train_id2id.append(labels[i].id)
       
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}
    color_2label = {label.color: label for label in labels}
    color_palette_train_ids = list(sum(color_palette_train_ids, ()))


    # mean: [  ]
    # std: [  ] 

    def __init__(self,
                    img_root: str,
                    label_root: str,
                    split: str,
                    switch_to_train_id: bool = False
                ) -> None:
        """
        Pascal Context dataset loader
        """
        self.root = img_root
        self.target_root = label_root
        self.split = split
        self.switch_to_train_id = switch_to_train_id

        # data root paths
        self.images_dir = os.path.join(self.root, self.split)
        self.targets_dir = os.path.join(self.target_root, self.split)

        self.images = []
        self.targets = []

        for root, dirs, files in os.walk(self.targets_dir):
            for file in files:
                img_name = f'{file.split(".")[0]}.jpg'
                
                self.images.append(os.path.join(self.images_dir, img_name))
                self.targets.append(os.path.join(self.targets_dir, file))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        np_label = np.uint8(np.array(target))
        if self.switch_to_train_id:
            # transform segmentation mask according to train_ids
            mask = 255*np.ones((np_label.shape), dtype=np.uint8)
            for i in np.unique(np_label):
                mask[np_label == i] = self.labels[i].train_id
            target = Image.fromarray(mask)

        path = self.images[index]
        
        return image, target, path

    def __len__(self) -> int:
        return len(self.images)

