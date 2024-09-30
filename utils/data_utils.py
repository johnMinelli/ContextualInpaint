import PIL
import cv2
import numpy as np
from PIL import Image


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
        patch_size = np.random.randint(50, 100)
        patch_shape = np.random.choice(['rectangle', 'circle'])
        patch_color = 255
        # Get patch position
        contour_points = contour[:, 0, :]
        random_index = np.random.randint(0, contour_points.shape[0])
        patch_position = contour_points[random_index]
        patch_x = patch_position[0] - patch_size // 2
        patch_y = patch_position[1] - patch_size // 2
        # Ensure the patch stays within the image boundaries
        patch_x = max(0, min(patch_x, width - patch_size))
        patch_y = max(0, min(patch_y, height - patch_size))
        # Draw
        if patch_shape == 'rectangle':
            mask[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size] = patch_color
        elif patch_shape == 'circle':
            radius = np.random.randint(0, patch_size)
            cv2.circle(mask, (patch_x, patch_y), radius, patch_color, -1)

    return mask


def mask_augmentation(mask_image, expansion_p=1., patch_p=1., min_expansion_factor=1., max_expansion_factor=1.6, patches=3):
    mask = np.array(mask_image) if isinstance(mask_image, PIL.Image.Image) else mask_image

    # Apply image segmentation to isolate the mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # Randomly expand the contour
    if np.random.random() < expansion_p:
        expansion_factor = np.random.uniform(min_expansion_factor, max_expansion_factor)
        expanded_contour = expand_contour(contour, expansion_factor)
        mask = cv2.drawContours(mask, [expanded_contour], 0, 255, -1)
    # Randomly apply additional patches
    if patches > 0 and np.random.random() < patch_p:
        num_patches = np.random.randint(1, patches)
        mask = apply_additional_patches(mask, num_patches, contour)

    mask = Image.fromarray(mask) if isinstance(mask_image, PIL.Image.Image) else mask
    return mask
