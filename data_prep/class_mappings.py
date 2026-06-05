import numpy as np

# 8-Class Common Definition
# Classes used across all experiments for evaluation and training
CLASSES_8CLASS = [
    "sky",          # 0
    "ground",       # 1
    "road",         # 2
    "construction", # 3
    "vegetation",   # 4
    "object",       # 5
    "vehicle",      # 6
    "human"         # 7
]

CLASS_NAMES_8CLASS = {i: name for i, name in enumerate(CLASSES_8CLASS)}
IGNORE_INDEX = 255

# RANUS (0-10) -> 8-class common (0-7)
# Original RANUS classes 1-10. 0 is ignore. Water(3) and Mountain(4) are mapped to ignore.
RANUS_TO_8CLASS_DICT = {
    0: 255, # Ignore
    1: 0,   # sky -> sky
    2: 1,   # ground -> ground
    3: 255, # water -> Ignore
    4: 255, # mountain -> Ignore
    5: 2,   # road -> road
    6: 3,   # construction -> construction
    7: 4,   # vegetation -> vegetation
    8: 5,   # object -> object
    9: 6,   # vehicle -> vehicle
    10: 7,  # human -> human
}

# Cityscapes/GTA5 (0-18) -> 8-class common (0-7)
CITYSCAPES_TO_8CLASS_DICT = {
    0: 2,   # road -> road
    1: 1,   # sidewalk -> ground
    2: 3,   # building -> construction
    3: 3,   # wall -> construction
    4: 3,   # fence -> construction
    5: 5,   # pole -> object
    6: 5,   # traffic light -> object
    7: 5,   # traffic sign -> object
    8: 4,   # vegetation -> vegetation
    9: 1,   # terrain -> ground
    10: 0,  # sky -> sky
    11: 7,  # person -> human
    12: 7,  # rider -> human
    13: 6,  # car -> vehicle
    14: 6,  # truck -> vehicle
    15: 6,  # bus -> vehicle
    16: 6,  # train -> vehicle
    17: 6,  # motorcycle -> vehicle
    18: 6,  # bicycle -> vehicle
    -1: 255,
    255: 255
}

def create_lookup_table(mapping_dict, max_val=256, default=255):
    """Creates a lookup table for fast mapping using numpy indexing."""
    lut = np.full((max_val,), default, dtype=np.uint8)
    for k, v in mapping_dict.items():
        if 0 <= k < max_val:
            lut[k] = v
    return lut

RANUS_8CLASS_LUT = create_lookup_table(RANUS_TO_8CLASS_DICT)
CITYSCAPES_8CLASS_LUT = create_lookup_table(CITYSCAPES_TO_8CLASS_DICT)

def map_ranus_to_8class(mask):
    """Maps RANUS mask directly to 8 common classes."""
    return RANUS_8CLASS_LUT[mask.astype(np.uint8)]

def map_cityscapes_to_8class(mask):
    """Maps Cityscapes mask to 8 common classes."""
    return CITYSCAPES_8CLASS_LUT[mask.astype(np.uint8)]
