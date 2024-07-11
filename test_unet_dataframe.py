import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from train_unet import dice_coefficient, dice_loss, encode_rle

from paths import *
from dataset_info import id_col, mask_col

def result_prediction(df, model, path_to_folder):
    """
    Function to generate predictions for images in a DataFrame and update the DataFrame with the encoded pixel masks.

    Args:
        df (pandas.DataFrame): DataFrame containing image IDs and associated information.
        model (keras.Model): Model used for predictions.
        path_to_folder (str): Path to the folder containing the images.

    Returns:
        pandas.DataFrame: DataFrame with updated predictions.
    """
    # Create a copy of the original DataFrame to store the updated predictions
    updated_df = df.copy()

    # Iterate over each row in the DataFrame `df` using its index
    for index, row in df.iterrows():
        # Get the path to the image from the 'ImageId' column in DataFrame `df`
        image_path = path_to_folder + row[id_col]
        # Read the image at the specified path and convert it to RGB format
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Add a dimension to create a batch of images
        image = np.expand_dims(image, axis=0)
        # Pass the image to the model and get the predicted masks
        predictions = model.predict(np.array(image))
        # Determine the threshold for binarizing the mask
        threshold = np.mean(predictions.squeeze())
        # Apply the threshold to the predicted values to obtain a binary mask
        mask = np.where(predictions.squeeze() > threshold, 1, 0)
        # Encode the mask into an RLE vector
        encoded_pixels = encode_rle(mask)
        # Write the encoded pixels to the corresponding 'EncodedPixels' column in the updated DataFrame
        updated_df.at[index, mask_col] = encoded_pixels
    print(updated_df)
    return updated_df

def main():
    # Define custom objects for U-net model
    tf.keras.utils.get_custom_objects()['dice_coefficient'] = dice_coefficient
    tf.keras.utils.get_custom_objects()['dice_loss'] = dice_loss

    # Loading the pre-trained U-net model
    model = load_model( filepath_keras )

    # Read the test data CSV file into a DataFrame
    test_data = pd.read_csv( test_dir_csv )

    # Perform predictions on the test data using the loaded model
    updated_df = result_prediction( test_data, model, test_dir_img )

    # Save the updated DataFrame with predictions to a CSV file
    updated_df.to_csv( output_file_path, index=False )

if __name__ == "__main__":
    main()