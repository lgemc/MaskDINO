# Copyright (c) Facebook, Inc. and its affiliates.
import os
import csv
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

# Dataset paths
HERDNET_ROOT = os.getenv("DETECTRON2_DATASETS", "datasets")

_HERDNET_CATEGORIES = [
    {"id": 1, "name": "Alcelaphinae"},
    {"id": 2, "name": "Buffalo"},
    {"id": 3, "name": "Kob"},
    {"id": 4, "name": "Warthog"},
    {"id": 5, "name": "Waterbuck"},
    {"id": 6, "name": "Elephant"},
]


def load_herdnet_csv(csv_file, image_root):
    """
    Load HerdNet dataset from CSV format.

    CSV format: images,labels,base_images,x_min,y_min,x_max,y_max
    """
    from PIL import Image
    dataset_dicts = {}

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row['images']
            image_path = os.path.join(image_root, image_name)

            if image_name not in dataset_dicts:
                # Get image dimensions
                img = Image.open(image_path)
                width, height = img.size

                dataset_dicts[image_name] = {
                    "file_name": image_path,
                    "image_id": image_name,
                    "height": height,
                    "width": width,
                    "annotations": []
                }

            x_min = float(row['x_min'])
            y_min = float(row['y_min'])
            x_max = float(row['x_max'])
            y_max = float(row['y_max'])
            category_id = int(row['labels'])

            # Create polygon mask from bounding box (rectangle)
            segmentation = [[
                x_min, y_min,
                x_max, y_min,
                x_max, y_max,
                x_min, y_max
            ]]

            annotation = {
                "bbox": [x_min, y_min, x_max, y_max],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": category_id - 1,  # Convert to 0-indexed
                "segmentation": segmentation,
            }
            dataset_dicts[image_name]["annotations"].append(annotation)

    return list(dataset_dicts.values())


def _get_herdnet_instances_meta():
    """
    Get metadata for HerdNet dataset
    """
    thing_ids = [k["id"] for k in _HERDNET_CATEGORIES]
    thing_colors = [
        [220, 20, 60],   # Alcelaphinae - red
        [119, 11, 32],   # Buffalo - dark red
        [0, 0, 142],     # Kob - blue
        [0, 0, 230],     # Warthog - light blue
        [106, 0, 228],   # Waterbuck - purple
        [0, 60, 100],    # Elephant - dark blue
    ]
    assert len(thing_ids) == 6, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in _HERDNET_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_herdnet_csv(name, metadata, csv_file, image_root):
    """
    Register a HerdNet dataset from CSV format for instance detection

    Args:
        name (str): the name that identifies a dataset, e.g. "herdnet_train"
        metadata (dict): extra metadata associated with this dataset
        csv_file (str): path to the CSV annotation file
        image_root (str): directory which contains all the images
    """
    DatasetCatalog.register(name, lambda: load_herdnet_csv(csv_file, image_root))
    MetadataCatalog.get(name).set(
        csv_file=csv_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def register_all_herdnet(root):
    """
    Register all HerdNet datasets from CSV format (1024x1024 patches)
    """
    csv_root = "/home/lmanrique/Do/HerdNetLGM/data/bbox/1024"

    for split in ["train", "val", "test"]:
        register_herdnet_csv(
            name=f"herdnet_{split}",
            metadata=_get_herdnet_instances_meta(),
            csv_file=os.path.join(csv_root, split, "gt.csv"),
            image_root=os.path.join(csv_root, split),
        )


# Register datasets
if __name__.endswith(".register_herdnet"):
    register_all_herdnet(HERDNET_ROOT)
