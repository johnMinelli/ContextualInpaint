import random

import cv2
import numpy as np


def not_image(filename: str):
    return not filename.lower().endswith(".jpeg") and not filename.lower().endswith(".jpg") and not filename.lower().endswith(".png")


def proc_collate_fn(data):
    # collated_data = default_collate(data)
    collated_data = {k:[b[k] for b in data] for k,v in data[0].items()}
    return collated_data


def expand_contour(contour, expansion_factor):
    # Compute the centroid of the contour
    moments = cv2.moments(contour)
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    # Expand the contour by a given factor around the centroid
    expanded_contour = contour.copy()
    expanded_contour[:, 0, 0] = cx + (contour[:, 0, 0] - cx) * expansion_factor
    expanded_contour[:, 0, 1] = cy + (contour[:, 0, 1] - cy) * expansion_factor

    return expanded_contour.astype(np.int32)


def apply_additional_patches(mask, num_patches, contour):
    height, width = mask.shape[:2]

    for _ in range(num_patches):
        patch_size = random.randint(50, 100)
        patch_shape = random.choice(['rectangle', 'circle'])
        patch_color = 255
        # Get patch position
        contour_points = contour[:, 0, :]
        patch_position = random.choice(contour_points)
        patch_x = patch_position[0] - patch_size // 2
        patch_y = patch_position[1] - patch_size // 2
        # Ensure the patch stays within the image boundaries
        patch_x = max(0, min(patch_x, width - patch_size))
        patch_y = max(0, min(patch_y, height - patch_size))
        # Draw
        if patch_shape == 'rectangle':
            mask[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size] = patch_color
        elif patch_shape == 'circle':
            radius = random.randint(0, patch_size)
            cv2.circle(mask, (patch_x, patch_y), radius, patch_color, -1)

    return mask


def mask_augmentation(mask_image):
    # Apply image segmentation to isolate the mask
    _, mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = random.choice(contours)

    # Randomly expand the contour
    if random.random() < 0.8:
        expansion_factor = random.uniform(1.1, 1.5)
        expanded_contour = expand_contour(contour, expansion_factor)
        mask = cv2.drawContours(mask, [expanded_contour], 0, 255, -1)
    # Randomly apply additional patches
    if random.random() < 0.5:
        num_patches = random.randint(1, 3)
        mask = apply_additional_patches(mask, num_patches, contour)

    return mask
