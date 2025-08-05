import numpy as np
import cv2

def paired_data_augmentation(input_image1, input_image2):
    # Stack the input images
    stacked_images = np.stack([input_image1, input_image2])

    # Random horizontal flip
    if np.random.rand() > 0.5:
        stacked_images = np.flip(stacked_images, axis=2)

    # Random vertical flip
    if np.random.rand() > 0.5:
        stacked_images = np.flip(stacked_images, axis=1)

    # Random rotation
    angle = np.random.uniform(-45, 45)  # Adjusted to the range -45 to 45 degrees
    rows, cols = input_image1.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    stacked_images = np.array([cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=cv2.INTER_CUBIC) for img in stacked_images])

    # Separate the augmented images back
    augmented_image1, augmented_image2 = stacked_images

    return augmented_image1, augmented_image2

def paired_data_augmentation_pet(input_image1, input_image2):
    # Stack the input images
    stacked_images = np.stack([input_image1, input_image2])

    # Random horizontal flip
    if np.random.rand() > 0.5:
        stacked_images = np.flip(stacked_images, axis=2)

    # Random vertical flip
    if np.random.rand() > 0.5:
        stacked_images = np.flip(stacked_images, axis=1)

    # Random 90 or -90 degree rotation
    if np.random.rand() > 0.5:
        angle = 90 if np.random.rand() > 0.5 else -90
        stacked_images = np.array([cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if angle == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE) for img in stacked_images])

    # Random zoom
    zoom_factor = np.random.uniform(1.01, 1.01)  # Zoom factor between 0.5 and 1.5
    stacked_images = np.array([cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC) for img in stacked_images])

    # Assuming the input images are square, else additional steps are needed to handle cropping after rotation
    # Random crop to original size
    rows, cols = input_image1.shape[:2]
    crop_x = np.random.randint(0, stacked_images.shape[2] - cols)
    crop_y = np.random.randint(0, stacked_images.shape[1] - rows)
    stacked_images = stacked_images[:, crop_y:crop_y+rows, crop_x:crop_x+cols]

    # Separate the augmented images back
    augmented_image1, augmented_image2 = stacked_images

    return augmented_image1, augmented_image2