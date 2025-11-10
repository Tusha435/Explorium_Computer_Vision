"""
Utilities package.
"""
from .coco_classes import (
    COCO_CLASSES,
    VEHICLE_CLASSES,
    TARGET_CLASSES,
    get_class_name,
    get_class_id,
    is_vehicle,
    is_target_class,
    get_class_color
)

__all__ = [
    'COCO_CLASSES',
    'VEHICLE_CLASSES',
    'TARGET_CLASSES',
    'get_class_name',
    'get_class_id',
    'is_vehicle',
    'is_target_class',
    'get_class_color'
]
