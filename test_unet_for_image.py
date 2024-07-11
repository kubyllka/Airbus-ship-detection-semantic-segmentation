from paths import *
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from train_unet import dice_loss, dice_coefficient

def display_image_and_mask(image_path, mask):
    """
    Function to display an original image and its corresponding segmented mask.

    Parameters:
        image_path (str): The file path of the original image.
        mask (numpy.ndarray): The segmented mask image.

    Returns:
        None

    """
    mask = np.array(mask * 255).astype('uint8')
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Concatenate the original image and the mask horizontally
    combined_image = np.hstack((image, mask))

    # Create a resizable window
    cv2.namedWindow('Image and mask', cv2.WINDOW_NORMAL)

    # Get the screen resolution
    screen_width, screen_height = 1920, 1080

    # Resize the window to fit the screen size
    cv2.resizeWindow('Image and mask', screen_width, screen_height)

    # Display the combined image
    cv2.imshow('Image and mask', combined_image)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define custom objects for U-net model
tf.keras.utils.get_custom_objects()['dice_coefficient'] = dice_coefficient
tf.keras.utils.get_custom_objects()['dice_loss'] = dice_loss

# Load pre-trained U-net model
model = load_model(filepath_keras)
# Read the image for testing
image = cv2.imread(image_path_test)
# OpenCV reads images in BGR format, so convert it to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255. # Image normalization
image = np.expand_dims(image, axis=0) # Batch dimension

# Predict the mask
predictions = model.predict(np.array(image))

# Calculate threshold for binary mask
threshold = np.mean(predictions.squeeze())

# Apply the threshold to the predictions
mask = np.where(predictions.squeeze() > threshold, 1, 0)

# Display original image and predicted mask
display_image_and_mask(image_path_test, mask)